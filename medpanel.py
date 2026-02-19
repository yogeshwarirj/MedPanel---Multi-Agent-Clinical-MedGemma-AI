"""
MedPanel - Multi-Agent AI Clinical Decision Support System
"""

import torch
import json
import re
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import faiss

# Global variables
processor = None
model = None
embed_model = None

def initialize_models(hf_token=None):
    """Initialize MedGemma and embedding models"""
    global processor, model, embed_model
    
    print("⏳ Loading MedGemma...")
    model_id = "google/medgemma-4b-it"
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    model.eval()
    print("✅ MedGemma loaded!")
    
    print("⏳ Loading PubMed embedding model...")
    Entrez.email = "medpanel@example.com"
    embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    print("✅ Models initialized!")


def call_medgemma(prompt, image=None, max_tokens=400):
    """Call MedGemma with proper format"""
    if image:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }]
    else:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    result = processor.decode(outputs[0], skip_special_tokens=True)
    
    if "model\n" in result:
        result = result.split("model\n")[-1].strip()
    
    return result


def safe_json(text):
    """Extract JSON from response"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        parsed = json.loads(text)
        return parsed
    except:
        pass
    
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"⚠️ JSON parse error: {e}")
    
    return {"raw_response": text}


def fetch_and_retrieve(query, top_k=3):
    """Fetch and retrieve relevant PubMed abstracts"""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=8)
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        raw = handle.read()
        abstracts = [b.strip() for b in raw.split("\n\n") if len(b.strip()) > 100]
        
        if not abstracts:
            return []

        embeddings = embed_model.encode(abstracts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        q_emb = embed_model.encode([query])
        _, idxs = index.search(np.array(q_emb), min(top_k, len(abstracts)))
        return [abstracts[i] for i in idxs[0]]
    except Exception as e:
        print(f"⚠️ PubMed error: {e}")
        return []


# Agent 1: Radiologist
def radiologist_agent(image, notes):
    if image is None:
        print("🩻 Radiologist (Skipped - No image)")
        return {
            "image_findings": [],
            "suspected_conditions": [],
            "abnormalities_detected": False,
            "note": "No image provided"
        }
    
    print("🩻 Radiologist analyzing...")
    prompt = f"""You are a radiologist. Analyze this chest X-ray.

Clinical notes: {notes}

Return JSON with: image_findings (list), suspected_conditions (list), abnormalities_detected (bool), severity (mild/moderate/severe), confidence (low/medium/high)"""
    
    result = call_medgemma(prompt, image)
    return safe_json(result)


# Agent 2: Internist
def internist_agent(notes):
    print("🩺 Internist analyzing...")
    prompt = f"""You are an internal medicine physician.

Patient symptoms: {notes}

Return JSON with: symptom_analysis (string), differential_diagnoses (list of 3), risk_factors (list), urgency (routine/urgent/emergency), confidence (low/medium/high)"""
    
    result = call_medgemma(prompt)
    return safe_json(result)


# Agent 3: Evidence Reviewer
def evidence_agent(radiology_out, internist_out):
    print("📚 Evidence Reviewer searching...")
    queries = (
        radiology_out.get("suspected_conditions", [])[:2] +
        internist_out.get("differential_diagnoses", [])[:2]
    )
    evidence = []
    for q in queries:
        if q and isinstance(q, str):
            results = fetch_and_retrieve(str(q), top_k=2)
            evidence.extend(results)
    print(f"   → Retrieved {len(evidence)} abstracts")
    return evidence[:4]


# Agent 4: Devil's Advocate
def devils_advocate_agent(image, notes, radiology_out, internist_out, evidence):
    print("😈 Devil's Advocate challenging...")
    evidence_text = "\n".join(evidence[:2]) if evidence else "No evidence"
    
    prompt = f"""Critical care specialist reviewing case.

Patient: {notes}
Radiologist: {radiology_out.get('suspected_conditions', [])}
Internist: {internist_out.get('differential_diagnoses', [])}
Evidence: {evidence_text[:500]}

What dangerous diagnoses might be missed? Return JSON with: missed_diagnoses (list), dangerous_alternatives (list), challenge_statement (string), requires_human_review (bool)"""
    
    result = call_medgemma(prompt, image)
    return safe_json(result)


# Orchestrator
def orchestrator_agent(notes, radiology_out, internist_out, evidence, devils_out):
    print("🎯 Orchestrator synthesizing...")
    
    prompt = f"""Synthesize medical panel review.

Radiologist: {json.dumps(radiology_out)}
Internist: {json.dumps(internist_out)}
Devil's Advocate: {json.dumps(devils_out)}
Evidence: {len(evidence)} abstracts

Return JSON with: primary_diagnosis (string), differential_diagnoses (list), panel_agreement_score (0-100), red_flags (list), recommended_next_steps (list), escalate_to_human (bool), escalation_reason (string or null), patient_summary (2-3 sentences)"""
    
    result = call_medgemma(prompt)
    return safe_json(result)


def run_medpanel(image, notes):
    """Run the complete MedPanel pipeline"""
    print("\n🏥 MedPanel Starting Panel Review...")
    print("="*60)
    
    trace = []
    
    r1 = radiologist_agent(image, notes)
    trace.append({"agent": "Radiologist", "output": r1})
    
    r2 = internist_agent(notes)
    trace.append({"agent": "Internist", "output": r2})
    
    evidence = evidence_agent(r1, r2)
    trace.append({"agent": "Evidence", "abstracts": len(evidence)})
    
    devil = devils_advocate_agent(image, notes, r1, r2, evidence)
    trace.append({"agent": "Devil's Advocate", "output": devil})
    
    final = orchestrator_agent(notes, r1, r2, evidence, devil)
    trace.append({"agent": "Orchestrator", "output": final})
    
    print("="*60)
    print("✅ Panel complete!\n")
    
    return {"panel_trace": trace, "final_report": final}
