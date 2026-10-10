"""Shared request plumbing for the LLM stages (generation, evaluation, semantic entropy).

* make_client: OpenAI-compatible client with a request timeout and the SDK's own retries off.
* chat: one chat completion with exponential backoff + jitter on 429 / 5xx / timeouts, and an
  immediate stop (FatalAPIError) on a rejected API key, so a bad key fails once, not per case.
* run_in_pool: runs one task per case on a thread pool and yields results to the main thread,
  which is the only thread that writes files. Ctrl+C cancels queued cases and waits for the
  in-flight ones so they can still be saved.
"""

import argparse
import os
import random
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import openai
from openai import OpenAI

DEFAULT_WORKERS = 8
REQUEST_TIMEOUT = 300  # seconds; a long arbitration answer from a slow model can take minutes
MAX_ATTEMPTS = 5
BACKOFF_BASE = 2   # seconds before the first retry, doubled each attempt
BACKOFF_CAP = 60
SAVE_EVERY = 10    # completed cases between CSV saves
RETRY_STATUS = {408, 409, 429}  # plus every 5xx

# Set once the run is stopping (Ctrl+C or a fatal error): workers give up instead of retrying.
STOP = threading.Event()


class FatalAPIError(Exception):
    """An error every other request would hit too (bad key, no credits), so the run stops."""


class RunStopped(Exception):
    """Raised in a worker once STOP is set; the case is left undone for --resume."""


class EmptyResponse(Exception):
    """A 200 response without choices or with empty content (OpenRouter does this when an upstream
    provider fails)."""


def default_workers(config):
    return int((config.get("concurrency") or {}).get("workers", DEFAULT_WORKERS))


def add_workers_arg(parser, config):
    def positive_int(value):
        if int(value) < 1:
            raise argparse.ArgumentTypeError("--workers must be at least 1")
        return int(value)
    default = default_workers(config)
    parser.add_argument("--workers", type=positive_int, default=default,
                        help=f"Maximum concurrent API requests (default {default}, from config.yml "
                             f"concurrency.workers if set, else {DEFAULT_WORKERS}). 1 = run sequentially.")


def make_client(base_url, api_key):
    # max_retries=0: chat() does the retrying, so the SDK must not retry underneath it as well
    return OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)


def _retry_after(error):
    try:
        return min(float(error.response.headers.get("retry-after")), 2 * BACKOFF_CAP)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def chat(client, limiter=None, **request):
    """client.chat.completions.create(**request) with retries. `limiter` (a semaphore) is held
    only while a request is in flight, not while backing off."""
    for attempt in range(MAX_ATTEMPTS):
        if STOP.is_set():
            raise RunStopped()
        try:
            with limiter or nullcontext():
                response = client.chat.completions.create(**request)
            if not response.choices:
                raise EmptyResponse(f"response had no choices: {getattr(response, 'error', None) or response}")
            choice = response.choices[0]
            if not choice.message.content:
                # Reasoning models (e.g. gpt-oss) sometimes return null content, typically when the
                # reasoning used up the output budget (finish_reason='length') or the provider failed
                raise EmptyResponse(f"response had no content (finish_reason={choice.finish_reason!r})")
            return response
        except openai.AuthenticationError as e:
            raise FatalAPIError(
                f"API key rejected by {client.base_url} (401 Unauthorized). The key is invalid or revoked: "
                "check GENERATION_API_KEY / JUDGE_API_KEY in .env (or api_key in config.yml).") from e
        except openai.APIStatusError as e:
            if e.status_code == 402:
                raise FatalAPIError(f"{client.base_url} returned 402 Payment Required: the account is out of credits.") from e
            if e.status_code not in RETRY_STATUS and e.status_code < 500:
                raise  # 400 and other client errors: retrying would give the same answer
            error = e
        except (openai.APIConnectionError, EmptyResponse) as e:  # APITimeoutError is an APIConnectionError
            error = e
        if attempt == MAX_ATTEMPTS - 1:
            raise error
        delay = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** attempt)
        delay = max(delay / 2 + random.uniform(0, delay / 2), _retry_after(error))
        print(f"  [RETRY {attempt + 1}/{MAX_ATTEMPTS - 1}] {type(error).__name__}: {error} (waiting {delay:.0f}s)")
        if STOP.wait(delay):
            raise RunStopped()


def _outcome(future):
    """(result, None) or (None, error) for a finished future, or None if it was stopped/cancelled."""
    try:
        return future.result(), None
    except FatalAPIError:
        raise
    except (RunStopped, CancelledError):
        return None
    except Exception as e:
        return None, e


def run_in_pool(items, task, workers):
    """Run task(item) for each item on `workers` threads and yield (item, result, error) on the
    calling thread as tasks finish. workers=1 runs inline, in order, with no threads.

    A FatalAPIError from any task cancels the rest and is re-raised. On Ctrl+C, queued tasks are
    cancelled, in-flight ones are still yielded as they finish (Ctrl+C again stops waiting), and
    KeyboardInterrupt is re-raised, so the caller should save in a `finally`."""
    if workers <= 1:
        for item in items:
            try:
                result = task(item)
            except RunStopped:
                continue
            except FatalAPIError:
                raise
            except Exception as e:
                yield item, None, e
                continue
            yield item, result, None
        return

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(task, item): item for item in items}
    finished = False
    try:
        try:
            for future in as_completed(futures):
                outcome = _outcome(future)
                if outcome:
                    yield (futures[future], *outcome)
        except KeyboardInterrupt:
            STOP.set()
            for future in futures:
                future.cancel()
            running = [f for f in futures if not f.done()]
            print(f"\n[INTERRUPTED] Cancelled queued cases; waiting for {len(running)} in-flight case(s) "
                  "to finish so they can be saved (Ctrl+C again to stop now)...", flush=True)
            try:
                for future in as_completed(running):
                    outcome = _outcome(future)
                    if outcome:
                        yield (futures[future], *outcome)
            except KeyboardInterrupt:
                pass
            raise KeyboardInterrupt
        finished = True
    finally:
        if not finished:
            STOP.set()
            for future in futures:
                future.cancel()
        pool.shutdown(wait=finished, cancel_futures=True)


def save_csv(df, path):
    """Write via a temp file so an interrupt mid-write never leaves a truncated CSV."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_cli(main):
    """Run a script's main(); turn a fatal API error or Ctrl+C into one clear message and exit
    without waiting on worker threads still blocked in a request (their cases were not saved,
    so --resume redoes them)."""
    try:
        main()
        return
    except FatalAPIError as e:
        print(f"\n[FATAL] {e}\nStopped the run. Cases finished before this were saved.")
        code = 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Finished cases were saved. Rerun with --resume to do the rest.")
        code = 130
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
