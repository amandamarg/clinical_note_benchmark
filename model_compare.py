# pip install openai>=1.51
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

MODEL = "o4-mini"  # strong tool-calling + reasoning; adjust per your account

SYSTEM_PROMPT = """
You are a clinical comparison agent. Compare Document B against Document A and identify ONLY
clinically significant differences (ignore phrasing/formatting).

Clinically significant = likely to affect clinical decision-making, safety, or billing-quality:
- Diagnoses (added/removed/changed specificity or acuity)
- Allergies (added/removed/changed severity)
- Medications (new/removed/changed dose/route/frequency, contraindications)
- Procedures, imaging, consults (added/removed)
- Labs/vitals that are abnormal or change assessment/plan
- Care plans (DNR status, follow-up, escalation, patient instructions with safety impact)
- Contraindications, interactions, critical history elements

Rules:
1) For each significant item that is in B but not in A -> call report_added_doc.
2) For each significant item that is in A but missing from B -> call report_missing_doc.
3) DO NOT call any tool for non-significant or purely stylistic differences.
4) Include precise, minimal evidence in each call (short snippet + section + codes if available).
5) Confidence is 0–1; severity is one of: low | moderate | high | critical.
6) If no clinically significant differences, reply with a brief text response and DO NOT call tools.
"""

tools = [
    {
        "type": "function",
        "name": "report_added_doc",
        "description": "Use ONLY for clinically significant content present in B but not in A.",
        "parameters": {
            "type": "object",
            "properties": {
                "clinical_concept": {"type": "string", "description": "Short title of the finding (e.g., 'New Dx: CHF', 'Increased Lisinopril dose')."},
                "category": {"type": "string", "description": "diagnosis|allergy|medication|procedure|imaging|lab|vital|plan|other"},
                    "severity": {"type": "string", "enum": ["low","moderate","high","critical"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "description": "Why this matters clinically; one sentence."},
                    "codes": {
                        "type": "object",
                        "description": "If available. Include only what you are confident in.",
                        "properties": {
                            "ICD10": {"type": "array", "items": {"type": "string"}},
                            "SNOMED": {"type": "array", "items": {"type": "string"}},
                            "RxNorm": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string", "description": "e.g., 'Assessment & Plan', 'Meds'"},
                            "snippet_B": {"type": "string", "description": "Minimal text from B supporting the finding."},
                            "offsets_B": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] character offsets in B if known"}
                        },
                        "required": ["section","snippet_B"]
                    }
            },
            "required": ["clinical_concept","category","severity","confidence","rationale","evidence"]
        }
    },
    {
        "type": "function",
        "name": "report_missing_doc",
        "description": "Use ONLY for clinically significant content present in A but missing from B.",
        "parameters": {
            "type": "object",
            "properties": {
                "clinical_concept": {"type": "string"},
                "category": {"type": "string", "description": "diagnosis|allergy|medication|procedure|imaging|lab|vital|plan|other"},
                "severity": {"type": "string", "enum": ["low","moderate","high","critical"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "codes": {
                    "type": "object",
                    "properties": {
                        "ICD10": {"type": "array", "items": {"type": "string"}},
                        "SNOMED": {"type": "array", "items": {"type": "string"}},
                        "RxNorm": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "evidence": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "snippet_A": {"type": "string", "description": "Minimal text from A supporting what is missing in B."},
                        "offsets_A": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] character offsets in A if known"}
                    },
                    "required": ["section","snippet_A"]
                }
            },
            "required": ["clinical_concept","category","severity","confidence","rationale","evidence"]
        }
    }
]

def compare_documents(doc_a: str, doc_b: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Compare Document B to Document A and act per the rules."},
                {"type": "input_text", "text": doc_a},
                {"type": "input_text", "text": doc_b},
            ]
        }
    ]

    # Make the call. Tool calls (if any) will appear in response.output with type='tool_call'.
    resp = client.responses.create(
        model=MODEL,
        input=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Handle tool calls
    added, missing = [], []
    # The Responses API can return a list of output items; iterate and execute tools accordingly.
    for item in resp.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            if item.name == "report_added_doc":
                added.append(args)
            elif item.name == "report_missing_doc":
                missing.append(args)

    # If there were no tool calls, capture any text the model returned
    text_output = "\n".join([i.content[0].text for i in resp.output if getattr(i, "type", None) == "message"])

    return {
        "added": added,
        "missing": missing,
        "text": text_output.strip() or None,
        "raw": resp  # keep for audit if needed
    }

# ----------------- Example usage -----------------
if __name__ == "__main__":
    A = """Patient: Jane Doe, 54F. Diagnoses: Type 2 diabetes; HTN.
Meds: Metformin 1000mg BID; Lisinopril 10mg daily.
Allergies: NKDA. Assessment: Stable. Plan: Continue meds; f/u 3 months."""
    B = """Patient: Jane Doe, 54F. Diagnoses: Type 2 diabetes; HTN; New CHF (NYHA II).
Meds: Metformin 1000mg BID; Lisinopril 20mg daily; Furosemide 20mg daily added.
Allergies: Penicillin (anaphylaxis). Assessment: Mild volume overload.
Plan: Start low-sodium diet; f/u 2 weeks; ED precautions for dyspnea."""

    result = compare_documents(A, B)
    print("ADDED in B:", json.dumps(result["added"], indent=2))
    print("MISSING from B:", json.dumps(result["missing"], indent=2))
    print("TEXT (if no sig diffs):", result["text"])
