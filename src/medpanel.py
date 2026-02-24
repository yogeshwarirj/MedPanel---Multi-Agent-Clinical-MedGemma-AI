# medpanel.py
# Core logic for the MedPanel multi-agent diagnostic system.
# This file contains all 4 agents + orchestrator + RAG pipeline.
# Imported by app.py which runs the Gradio interface on HuggingFace Spaces.

import os
import json
import re
import torch
import numpy as np
import faiss

from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer
from Bio import Entrez
from PIL import Image


# ── Model Configuration ──────────────────────────────────────────────
# We load these once at startup so they're ready for every request
MODEL_ID = "google/medgemma-4b-it"

# NCBI requires an email for PubMed access — just for identification purposes
Entrez.email = "medpanel@example.com"


# ── Load Models ──────────────────────────────────────────────────────

def load_models():
    """
    Loads MedGemma and the PubMed embedding model into memory.
    Called once when the app starts up on HuggingFace Spaces.
    Returns processor, model, and embed_model.
    """

    print("Loading MedGemma model...")

    # Load the processor — handles both text tokenization and image preprocessing
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN")
        
    )

    # Load MedGemma in bfloat16 to fit within GPU memory limits
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
         low_cpu_mem_usage=True,         
        attn_implementation="eager"      
    )
    model.eval()
    print("✅ MedGemma loaded!")

    # Load the PubMed-specific embedding model for semantic search
    print("Loading PubMed embedding model...")
    embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    print("✅ Embedding model loaded!")

    return processor, model, embed_model


# Initialize all models at module load time
processor, model, embed_model = load_models()


# ── Base Caller ──────────────────────────────────────────────────────

def call_medgemma(prompt, image=None, max_tokens=400):
    """
    Sends a prompt (and optional image) to MedGemma and returns the response.
    This is the single point of contact with the model for all agents.
    """

    # Build message in MedGemma's expected chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *([{"type": "image", "image": image}] if image else [])
            ]
        }
    ]

    # Tokenize and move to the same device as the model
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response — no_grad saves memory, do_sample=False is deterministic
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    # Decode and strip the echoed prompt — we only want the model's reply
    full_response = processor.decode(output_tokens[0], skip_special_tokens=True)
    return full_response.split("model\n")[-1].strip()


def safe_json(text):
    """
    Safely extracts a JSON object from the model's response.
    Handles markdown code fences, extra text, and malformed JSON.
    Always returns a dict — never crashes.
    """

    # Strip markdown fences like ```json ... ``` if present
    for fence_start, fence_end in [("```json", "```"), ("```", "```")]:
        if fence_start in text:
            text = text.split(fence_start)[1].split(fence_end)[0].strip()
            break

    # Try standard JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to regex — find any { ... } block in the response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    try:
        return json.loads(json_match.group()) if json_match else {"raw_response": text}
    except json.JSONDecodeError:
        return {"raw_response": text}


# ── PubMed RAG ───────────────────────────────────────────────────────

def fetch_and_retrieve(query, top_k=3):
    """
    Searches PubMed for relevant abstracts using the given query.
    Uses FAISS + PubMedBERT embeddings to find the most semantically
    similar abstracts rather than just keyword matching.
    Returns a list of abstract strings.
    """

    try:
        # Search PubMed for matching paper IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=8)
        ids = Entrez.read(handle)["IdList"]

        if not ids:
            return []

        # Fetch the actual abstract text for those papers
        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="text"
        )

        # Split the bulk text into individual abstracts, filter out short chunks
        raw_text = handle.read()
        abstracts = [
            chunk.strip()
            for chunk in raw_text.split("\n\n")
            if len(chunk.strip()) > 100
        ]

        if not abstracts:
            return []

        # Build FAISS index from abstract embeddings
        embeddings = embed_model.encode(abstracts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        # Find the top_k most relevant abstracts for our query
        query_embedding = embed_model.encode([query])
        _, best_indices = index.search(
            np.array(query_embedding),
            min(top_k, len(abstracts))
        )

        return [abstracts[i] for i in best_indices[0]]

    except Exception as e:
        # If PubMed is unavailable, return empty rather than crashing
        print(f"PubMed fetch failed for '{query}': {e}")
        return []


# ── Agent 1: Radiologist ─────────────────────────────────────────────

def radiologist_agent(image, notes):
    """
    Analyzes the medical image and returns structured radiology findings.
    If no image is provided, returns a safe empty result.
    """

    if not image:
        return {
            "suspected_conditions": [],
            "note": "No image provided — skipping radiology analysis"
        }

    # Convert to RGB if the image is grayscale — MedGemma requires RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    prompt = f"""You are an experienced radiologist reviewing a medical image.
Patient clinical notes: {notes}
Carefully analyze the image and return your findings as a JSON object with:
- image_findings: list of observed features (e.g. "upper lobe opacity")
- suspected_conditions: list of possible diagnoses based on what you see
- abnormalities_detected: true or false
- severity: one of "mild", "moderate", "severe", or "normal"
- confidence: your confidence level from 0.0 to 1.0
Return only the JSON object, no extra explanation."""

    return safe_json(call_medgemma(prompt, image))


# ── Agent 2: Internist ───────────────────────────────────────────────

def internist_agent(notes):
    """
    Analyzes clinical notes as an internal medicine physician.
    Returns differential diagnoses, risk factors, and urgency level.
    Works from text only — no image.
    """

    prompt = f"""You are an experienced internal medicine physician.
Patient clinical notes: {notes}
Based on the symptoms and clinical details, return your assessment as a JSON object with:
- differential_diagnoses: list of 3 most likely diagnoses, ordered by likelihood
- risk_factors: list of relevant patient risk factors
- urgency: one of "routine", "urgent", or "emergent"
- confidence: your overall confidence from 0.0 to 1.0
Return only the JSON object, no extra explanation."""

    return safe_json(call_medgemma(prompt))


# ── Agent 3: Evidence Reviewer ───────────────────────────────────────

def evidence_agent(r1, r2):
    """
    Fetches supporting medical literature from PubMed based on what
    the Radiologist and Internist suspected.
    Returns up to 4 relevant abstracts.
    """

    # Combine top conditions from both agents into search queries
    queries = (
        r1.get("suspected_conditions", [])[:2] +
        r2.get("differential_diagnoses", [])[:2]
    )

    # Search PubMed for each condition and collect abstracts
    evidence = []
    for query in queries:
        results = fetch_and_retrieve(str(query), top_k=2)
        evidence.extend(results)

    # Cap at 4 to avoid overflowing the model's context window
    return evidence[:4]


# ── Agent 4: Devil's Advocate ────────────────────────────────────────

def devils_advocate_agent(image, notes, r1, r2, evidence):
    """
    Adversarial agent that challenges the other agents' conclusions.
    Specifically looks for dangerous diagnoses that were missed.
    This is the agent that catches TB when base MedGemma misses it.
    """

    # Short evidence snippet so we don't overflow the prompt
    evidence_snippet = "\n".join(evidence[:2]) if evidence else "None available"

    prompt = f"""You are a critical care specialist and patient safety advocate.
Your job is NOT to agree — your job is to find what everyone else missed.
Patient clinical notes: {notes}
The radiologist suspected: {r1.get('suspected_conditions', [])}
The internist concluded:   {r2.get('differential_diagnoses', [])}
Relevant medical literature:
{evidence_snippet[:500]}
Challenge these conclusions. Look for dangerous diagnoses that were missed,
rare but life-threatening alternatives, and overlooked red flags.
Return a JSON object with:
- missed_diagnoses: list of diagnoses the other agents may have overlooked
- dangerous_alternatives: list of serious conditions that must be ruled out
- challenge_statement: one sentence explaining your biggest concern
- requires_human_review: true or false
Return only the JSON object, no extra explanation."""

    # Pass image if available so the devil's advocate can see it too
    if image and image.mode != "RGB":
        image = image.convert("RGB")

    return safe_json(call_medgemma(prompt, image))


# ── Orchestrator ─────────────────────────────────────────────────────

def orchestrator_agent(notes, r1, r2, evidence, devil):
    """
    Synthesizes all four agents' outputs into a single final report.
    Decides on the primary diagnosis, confidence, escalation, and next steps.
    """

    prompt = f"""You are the lead physician synthesizing a multi-specialist panel review.
RADIOLOGIST findings:
{json.dumps(r1, indent=2)}
INTERNIST findings:
{json.dumps(r2, indent=2)}
DEVIL'S ADVOCATE concerns:
{json.dumps(devil, indent=2)}
Supporting evidence: {len(evidence)} PubMed abstracts retrieved.
Synthesize everything into a final clinical report as a JSON object with:
- primary_diagnosis: the single most likely diagnosis
- differential_diagnoses: list of other possibilities
- panel_agreement_score: 0-100, how much the specialists agreed
- red_flags: list of warning signs needing immediate attention
- recommended_next_steps: list of tests or actions to take
- escalate_to_human: true if a real doctor needs to review this urgently
- escalation_reason: why escalation is needed (or "Not required")
- patient_summary: 2-sentence plain English summary for the patient
Return only the JSON object, no extra explanation."""

    return safe_json(call_medgemma(prompt))


# ── Master Pipeline ──────────────────────────────────────────────────

def run_medpanel(image, notes):
    """
    Runs the full MedPanel multi-agent pipeline.
    Accepts a PIL image (or None) and a string of clinical notes.
    Returns a dict with panel_trace (each agent's output) and final_report.
    """

    trace = []

    # Step 1: Radiologist — analyze the image
    print("🩻 Running Radiologist agent...")
    r1 = radiologist_agent(image, notes)
    trace.append({"agent": "Radiologist", "output": r1})

    # Step 2: Internist — analyze the clinical notes
    print("🩺 Running Internist agent...")
    r2 = internist_agent(notes)
    trace.append({"agent": "Internist", "output": r2})

    # Step 3: Evidence Reviewer — fetch PubMed literature
    print("📚 Fetching PubMed evidence...")
    evidence = evidence_agent(r1, r2)
    trace.append({"agent": "Evidence Reviewer", "abstracts_retrieved": len(evidence)})

    # Step 4: Devil's Advocate — challenge the findings
    print("😈 Running Devil's Advocate agent...")
    devil = devils_advocate_agent(image, notes, r1, r2, evidence)
    trace.append({"agent": "Devil's Advocate", "output": devil})

    # Step 5: Orchestrator — synthesize the final report
    print("🏥 Synthesizing final report...")
    final_report = orchestrator_agent(notes, r1, r2, evidence, devil)
    trace.append({"agent": "Orchestrator", "output": final_report})

    print("✅ MedPanel analysis complete!")

    return {
        "panel_trace": trace,
        "final_report": final_report
    }
