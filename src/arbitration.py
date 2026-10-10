"""Pure prompt-building logic for the LLM arbitrator, shared by generation_pipeline.py and
semantic_entropy.py. No config, client or file I/O here, so importing this module has no
side effects.
"""

import inspect
import re

from signature_definitions import get_signature_reference_text

SYSTEM_INSTRUCTION = inspect.cleandoc("""
    You are an advanced bioinformatics AI specialising in genomic oncology.
    Your task is to analyse transcriptomic profiles, assess the prognostic risk classifications of several multi-gene
    signatures, explain any conflicts between them, and give a final risk classification.

    CRITICAL ANTI-HALLUCINATION CONSTRAINT:
    Base your biological synthesis EXCLUSIVELY on data explicitly provided in this prompt: the "Signature Definitions",
    and, when they are present, the "Clinical Metadata", the "Transcriptomic Profile" values and the "Verified Active
    Biological Pathways" context. You may reason about any gene listed in the Transcriptomic Profile using its given
    expression value, even if that gene did not trigger an entry in the Verified Active Biological Pathways section.
    Do not introduce genes, biomarkers, or pathways that are not explicitly present in the provided context.
    Describe how a named signature (e.g. Oncotype DX, PAM50, BCI) weights biomarkers only as stated in the
    Signature Definitions; do not invent other weights or gene memberships.
    If a section of evidence is not provided, do not make claims that would require it.
    Refer to the patient neutrally as "the patient". Do not state the patient's sex, receptor status (ER, PR, HER2)
    or molecular subtype unless it appears in the provided context.
    Do not recommend medical treatments or clinical therapies; focus strictly on prognostic risk classification.
""")

# Which evidence blocks the model sees. Signature classifications are always included.
ABLATIONS = {
    "full":          {"clinical": True,  "expression": True,  "pathways": True},
    "no_pathways":   {"clinical": True,  "expression": True,  "pathways": False},
    "clinical_only": {"clinical": True,  "expression": False, "pathways": False},
    "classes_only":  {"clinical": False, "expression": False, "pathways": False},
}

# Shown with the signature classes in every prompt (and so in the semantic-entropy evidence too)
CLASS_LEGEND = (
    "Each signature places the patient in Low, Intermediate or High risk. Each line gives the class, the "
    "patient's score and the cutoffs that produced the class. Intermediate means that signature does not take "
    "a side; it supports neither High nor Low risk."
)

FINAL_RESOLUTION_PATTERN = re.compile(r"FINAL RESOLUTION:\s*\**\s*(High|Low)\s+Risk", re.IGNORECASE)

def parse_final_resolution(text):
    """Return 'High Risk' or 'Low Risk' from the last FINAL RESOLUTION line, or None if absent."""
    matches = FINAL_RESOLUTION_PATTERN.findall(text or "")
    if not matches:
        return None
    return f"{matches[-1].capitalize()} Risk"

def majority_vote(classes):
    """High against Low calls across signature classes, ignoring Intermediate: 'High Risk',
    'Low Risk', or 'Tie' when the counts are equal (including all Intermediate)."""
    classes = list(classes)
    n_high, n_low = classes.count("High"), classes.count("Low")
    if n_high == n_low:
        return "Tie"
    return "High Risk" if n_high > n_low else "Low Risk"

def build_prompt(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation="full"):
    include = ABLATIONS[ablation]
    sections = [
        "Please act as a Bioinformatics Arbitrator and assess the prognostic risk classifications for this patient.",
        f"Patient ID: {patient_id}",
    ]
    if include["clinical"]:
        sections.append(f"--- Clinical Metadata ---\n{clinical_data}")
    # Static description of the algorithms; shown in every ablation because it is not patient data
    sections.append(f"--- Signature Definitions (how each algorithm scores a patient) ---\n{get_signature_reference_text()}")
    sections.append(
        "--- Algorithmic Risk Classifications (Low / Intermediate / High) ---\n"
        f"{CLASS_LEGEND}\n{signature_classifications}"
    )
    if include["expression"]:
        sections.append(f"--- Transcriptomic Profile (Key Biomarkers & Expression Levels) ---\n{transcriptomic_data}")
    if include["pathways"]:
        sections.append(f"--- Verified Active Biological Pathways (Reactome Database Extraction) ---\n{active_pathways}")

    evidence = "the Signature Definitions and the provided pathways" if include["pathways"] else "the Signature Definitions and the provided evidence"
    sections.append(
        "REQUIRED ARBITRATION STEPS:\n"
        "1. Classification Assessment: Review each algorithm's classification and state whether any conflict: a "
        "conflict is at least one High against at least one Low (Intermediate signatures do not take a side). "
        "If there is no conflict, say so.\n"
        f"2. Mechanistic Explanation: Using {evidence}, explain why the algorithms agree or disagree, based on how "
        "they mathematically weight different biomarkers and on this patient's values.\n"
        "3. Prognostic Resolution: Give a final risk classification based on the biological evidence provided."
    )
    sections.append(
        "IMPORTANT: Your final classification must be binary even if some signatures are Intermediate. "
        "Conclude your text with exactly one of these phrases:\n"
        "FINAL RESOLUTION: High Risk\n"
        "or\n"
        "FINAL RESOLUTION: Low Risk"
    )
    return "\n\n".join(sections)
