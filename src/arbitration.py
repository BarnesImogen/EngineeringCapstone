"""Pure prompt-building logic for the LLM arbitrator, shared by generation_pipeline.py and
semantic_entropy.py. No config, client or file I/O here, so importing this module has no
side effects.
"""

import inspect
import re

from signature_definitions import get_signature_reference_text

SYSTEM_INSTRUCTION = inspect.cleandoc("""
    You are an advanced bioinformatics AI specialising in genomic oncology.
    Your task is to analyse transcriptomic profiles and resolve discordant prognostic risk classifications across multi-gene signatures.

    CRITICAL ANTI-HALLUCINATION CONSTRAINT:
    Base your biological synthesis EXCLUSIVELY on data explicitly provided in this prompt: the "Signature Definitions",
    and, when they are present, the "Clinical Metadata", the "Transcriptomic Profile" values and the "Verified Active
    Biological Pathways" context. You may reason about any gene listed in the Transcriptomic Profile using its given
    expression value, even if that gene did not trigger an entry in the Verified Active Biological Pathways section.
    Do not introduce genes, biomarkers, or pathways that are not explicitly present in the provided context.
    Describe how a named signature (e.g. Oncotype DX, PAM50, BCI) weights biomarkers only as stated in the
    Signature Definitions; do not invent other weights or gene memberships.
    If a section of evidence is not provided, do not make claims that would require it.
    Do not recommend medical treatments or clinical therapies; focus strictly on prognostic risk classification.
""")

# Which evidence blocks the model sees. Signature classifications are always included.
ABLATIONS = {
    "full":          {"clinical": True,  "expression": True,  "pathways": True},
    "no_pathways":   {"clinical": True,  "expression": True,  "pathways": False},
    "clinical_only": {"clinical": True,  "expression": False, "pathways": False},
    "classes_only":  {"clinical": False, "expression": False, "pathways": False},
}

FINAL_RESOLUTION_PATTERN = re.compile(r"FINAL RESOLUTION:\s*\**\s*(High|Low)\s+Risk", re.IGNORECASE)

def parse_final_resolution(text):
    """Return 'High Risk' or 'Low Risk' from the last FINAL RESOLUTION line, or None if absent."""
    matches = FINAL_RESOLUTION_PATTERN.findall(text or "")
    if not matches:
        return None
    return f"{matches[-1].capitalize()} Risk"

def build_prompt(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation="full"):
    include = ABLATIONS[ablation]
    sections = [
        "Please act as a Bioinformatics Arbitrator to resolve the prognostic discordance for this patient.",
        f"Patient ID: {patient_id}",
    ]
    if include["clinical"]:
        sections.append(f"--- Clinical Metadata ---\n{clinical_data}")
    # Static description of the algorithms; shown in every ablation because it is not patient data
    sections.append(f"--- Signature Definitions (how each algorithm scores a patient) ---\n{get_signature_reference_text()}")
    sections.append(
        "--- Conflicting Algorithmic Risk Classifications (1 = High Risk, 0 = Low Risk) ---\n"
        f"{signature_classifications}"
    )
    if include["expression"]:
        sections.append(f"--- Transcriptomic Profile (Key Biomarkers & Expression Levels) ---\n{transcriptomic_data}")
    if include["pathways"]:
        sections.append(f"--- Verified Active Biological Pathways (Reactome Database Extraction) ---\n{active_pathways}")

    evidence = "the Signature Definitions and the provided pathways" if include["pathways"] else "the Signature Definitions and the provided evidence"
    sections.append(
        "REQUIRED ARBITRATION STEPS:\n"
        "1. Conflict Diagnosis: Explicitly state WHICH algorithms are conflicting.\n"
        f"2. Mechanistic Root Cause: Using {evidence}, explain EXACTLY why the algorithms disagreed based on how they mathematically weight different biomarkers.\n"
        "3. Prognostic Resolution: Deliver a final, tie-breaking risk classification based on the biological evidence provided."
    )
    sections.append(
        "IMPORTANT: Conclude your text with this exact phrase:\n"
        "FINAL RESOLUTION: High Risk\n"
        "or\n"
        "FINAL RESOLUTION: Low Risk"
    )
    return "\n\n".join(sections)
