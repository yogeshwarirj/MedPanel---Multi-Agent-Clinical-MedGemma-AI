# MedPanel Architecture

## System Overview

MedPanel is a multi-agent adversarial AI system for clinical decision support, built on Google's MedGemma-4B foundation model.

## Core Components

### 1. Base Model
- **Model:** google/medgemma-4b-it (MedGemma-4B Instruction-Tuned)
- **Type:** Multimodal (image + text)
- **Quantization:** bfloat16
- **Deployment:** HuggingFace Spaces (Gradio)

### 2. Five Specialized Agents

Each agent is an independent MedGemma instance with specialized prompting:

**Agent 1: Radiologist**
- **Input:** Medical image + clinical context
- **Role:** Analyze imaging findings
- **Output:** Structured JSON with findings, suspected conditions, severity

**Agent 2: Internist**
- **Input:** Clinical notes only (no image)
- **Role:** Analyze symptoms and presentation
- **Output:** Differential diagnoses with reasoning

**Agent 3: Evidence Reviewer**
- **Input:** Suspected conditions from Agents 1 & 2
- **Process:** 
  1. Query PubMed via NCBI Entrez API
  2. Embed abstracts with PubMedBERT
  3. Semantic search with FAISS (top-4 results)
- **Output:** Relevant medical literature

**Agent 4: Devil's Advocate** (KEY INNOVATION)
- **Input:** All prior agent outputs + clinical case
- **Role:** **Challenge conclusions and find missed diagnoses**
- **Prompt:** "What dangerous diagnoses might they have missed?"
- **Output:** Alternative diagnoses, concerns, red flags

**Agent 5: Orchestrator**
- **Input:** All agent outputs
- **Role:** Synthesize final diagnosis
- **Logic:**
  - Calculate panel agreement (0-100%)
  - Flag disagreements
  - **Escalation decision:** If agreement < 80% OR Devil's Advocate raised concerns → Escalate to human
- **Output:** Final report with diagnosis, confidence, escalation

### 3. RAG Pipeline

**Components:**
- **Data Source:** PubMed (NCBI Entrez API)
- **Embedding Model:** pritamdeka/S-PubMedBert-MS-MARCO
- **Vector Store:** FAISS (CPU)
- **Retrieval:** Top-4 most relevant abstracts per query

**Flow:**
1. Extract suspected conditions from agents
2. Query PubMed: `condition + "diagnosis" + "symptoms"`
3. Fetch abstracts (limit 10 per query)
4. Embed with PubMedBERT (768-dim vectors)
5. Store in FAISS index
6. Semantic search for top-4 most relevant
7. Pass to Devil's Advocate and Orchestrator

## Data Flow
```
┌─────────────────┐
│  Patient Input  │
│  (image + notes)│
└────────┬────────┘
         │
    ┌────▼─────────────────────────────────┐
    │  Agent 1: Radiologist                │
    │  → analyze_image(img, notes)         │
    └────┬─────────────────────────────────┘
         │ findings
    ┌────▼─────────────────────────────────┐
    │  Agent 2: Internist                  │
    │  → analyze_symptoms(notes)           │
    └────┬─────────────────────────────────┘
         │ differential_dx
    ┌────▼─────────────────────────────────┐
    │  Agent 3: Evidence Reviewer          │
    │  → search_pubmed(conditions)         │
    │  → retrieve_with_faiss(query, k=4)   │
    └────┬─────────────────────────────────┘
         │ evidence_abstracts
    ┌────▼─────────────────────────────────┐
    │  Agent 4: Devil's Advocate           │
    │  → challenge(all_outputs, case)      │
    │  → flag_dangerous_misses()           │
    └────┬─────────────────────────────────┘
         │ challenges + alternatives
    ┌────▼─────────────────────────────────┐
    │  Agent 5: Orchestrator               │
    │  → synthesize(all_agents)            │
    │  → calculate_agreement()             │
    │  → decide_escalation()               │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │  Final Report                        │
    │  • Primary diagnosis                 │
    │  • Panel agreement score (0-100%)    │
    │  • Differential diagnoses            │
    │  • Red flags                         │
    │  • Escalate to human (bool)          │
    │  • Escalation reason                 │
    └──────────────────────────────────────┘
```

## Technical Implementation

### Core Function
```python
def run_medpanel(image, notes):
    """
    Run full MedPanel workflow
    
    Args:
        image: PIL Image or None
        notes: str - Clinical notes
        
    Returns:
        dict with:
            - panel_trace: List of all agent outputs
            - final_report: Orchestrator synthesis
    """
    
    # Agent 1: Radiologist
    r1 = radiologist_agent(image, notes) if image else None
    
    # Agent 2: Internist
    r2 = internist_agent(notes)
    
    # Agent 3: Evidence Reviewer
    conditions = extract_conditions(r1, r2)
    r3 = evidence_agent(conditions)
    
    # Agent 4: Devil's Advocate
    r4 = devils_advocate_agent(image, notes, r1, r2, r3)
    
    # Agent 5: Orchestrator
    r5 = orchestrator_agent(r1, r2, r3, r4, notes)
    
    return {
        "panel_trace": [r1, r2, r3, r4],
        "final_report": r5
    }
```

### Escalation Logic
```python
def should_escalate(panel_outputs):
    """
    Decide if human review is needed
    
    Triggers:
    - Panel agreement < 80%
    - Devil's Advocate raised concerns
    - High-risk condition detected
    """
    
    agreement = calculate_agreement(panel_outputs)
    devils_concerns = panel_outputs[3].get("missed_diagnoses", [])
    
    if agreement < 80:
        return True, "Low panel agreement"
    
    if devils_concerns:
        return True, f"Devil's Advocate flagged: {devils_concerns}"
    
    return False, None
```

## Deployment

**Platform:** HuggingFace Spaces  
**Framework:** Gradio 4.44.0  
**GPU:** T4 (free tier)  
**Latency:** 30-60 seconds per case  
**Concurrency:** 1 (free tier limit)  

**Production Scaling:**
- Upgrade to dedicated GPU (A100)
- Enable request queuing
- Add response caching
- Target latency: <15 seconds

## Performance Characteristics

- **Memory:** ~8GB VRAM (model + embeddings)
- **Compute:** 5 sequential LLM calls + 1 RAG retrieval
- **Bottleneck:** Sequential agent calls (future: parallelize Radiologist + Internist)
- **Cost:** ~$0.02 per diagnosis (at scale with dedicated GPU)

## Future Enhancements

1. **Fine-tuning:** Specialize each agent on domain-specific data
2. **Parallel Execution:** Run Radiologist + Internist concurrently
3. **Streaming:** Show agent outputs in real-time
4. **Model Distillation:** Reduce latency to <10 seconds
5. **Confidence Calibration:** Improve agreement score accuracy
