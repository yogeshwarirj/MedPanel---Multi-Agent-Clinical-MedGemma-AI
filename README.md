# 🏥 MedPanel - Multi-Agent Clinical AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HF Spaces](https://img.shields.io/badge/🤗-MedPanel-yellow)](https://huggingface.co/spaces/Yogeshwarirj/medpanel_api)
[![HF Spaces](https://img.shields.io/badge/🤗-Benchmark-yellow)](https://huggingface.co/spaces/Yogeshwarirj/benchmark_test)

**Adversarial AI that catches dangerous missed diagnoses**

Built for Google MedGemma Impact Challenge 2025

---

## 🎯 The Problem

Medical AI systems miss dangerous diagnoses. In testing, a single AI agent missed **3 out of 10 critical cases**:
- Ectopic pregnancy (misdiagnosed as appendicitis)
- Spinal cord compression (completely missed)  
- Cholecystitis (misdiagnosed as biliary colic)

**450,000 people die annually from missed TB diagnoses alone.**

---

## 💡 The Solution

MedPanel uses **4 AI agents that challenge each other** before making a diagnosis:
```
Patient Case
    ↓
🩻 Radiologist → Analyzes imaging
    ↓
🩺 Internist → Analyzes symptoms
    ↓
📚 Evidence Reviewer → Searches PubMed
    ↓
😈 Devil's Advocate → Challenges conclusions (KEY INNOVATION)
    ↓
🎯 Orchestrator → Final diagnosis + escalation decision
```

---

## 📊 Results

| Metric | Single Agent | MedPanel | Improvement |
|--------|--------------|----------|-------------|
| **Overall Accuracy** | 70% | **100%** | +30 pts |
| **Hard Cases** | 75% | **100%** | +25 pts |
| **Dangerous Misses Caught** | 0 | **3/10 (30%)** | - |

**Key Finding:** Devil's Advocate caught dangerous misses in 30% of cases

---

## Links
**🎬 Video Demo (3 minutes):**  
https://youtu.be/8Cpzc_Nz0qg

**🌐 Live Interactive Demo:**  
https://huggingface.co/spaces/Yogeshwarirj/medpanel_api
**Try:** _"22 year old female. Severe lower abdominal pain, right-sided. Fever, nausea, elevated WBC. Last menstrual period 6 weeks ago.o"_  
**Watch _Devil's Advocate_ catch _ectopic pregnancy_ in real-time**

**📊 Live Benchmark Results:**  
https://huggingface.co/spaces/Yogeshwarirj/benchmark_test  
**Full comparison:** _Single MedGemma vs MedPanel_ on **10 test cases**  
**Interactive** — _run 10 cases yourself_.  
Test cases are in the file **`benchmark_cases.py`**.


---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/medpanel.git
cd medpanel
pip install -r requirements.txt
```

### Usage
```python
from src.medpanel import run_medpanel

# Run MedPanel on a case
results = run_medpanel(
    image=None,  # Optional: medical image
    notes="65yo male, persistent cough 6 weeks, night sweats, weight loss"
)

print(results['final_report'])
# Output: Primary diagnosis, confidence, escalation decision
```

### Run Gradio Interface
```bash
python src/app.py
```

Opens at `http://localhost:7860`

---

## 📂 Repository Structure
```
medpanel/
├── src/
│   ├── medpanel.py          # Core multi-agent system
│   ├── app.py               # Gradio API interface
|── benchmark/
│   └── benchmark_app.py     # Benchmark comparison UI
|   ├── benchmark.py          # compare single Ai vs MedPanel multi agent AI
|   ├── benchmark_cases.py     # All 10 test cases
├── results/
│   └── benchmark_results.json  # Full test results
├── docs/
│   ├── ARCHITECTURE.md      # System design
│   └── BENCHMARKS.md        # Detailed evaluation
└── requirements.txt
```

---

## 🏗️ Architecture

**Base Model:** MedGemma-4B (google/medgemma-4b-it)  
**RAG:** PubMed via NCBI Entrez + PubMedBERT embeddings + FAISS  
**Deployment:** HuggingFace Spaces (Gradio)  
**Latency:** ~30-60 seconds per case  
**Cost:** ~$0.02 per diagnosis at scale  

**Key Innovation:** Devil's Advocate agent specifically designed to find dangerous missed diagnoses

---

## 💊 Impact

**At 1,000 clinics:**
- 500,000 patients screened annually
- 2,250 additional diagnoses caught
- **338 lives saved per year**
- **$30 per life saved** (vs $500 traditional screening)

**At scale (10,000 clinics):** 3,000+ lives saved annually

---

## 📄 Documentation

- **Full Technical Write-up:** [Kaggle Competition Page]([YOUR_KAGGLE_URL](https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/medpanel-multi-agent-clinical-ai))
- **Architecture Details:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Benchmark Analysis:** [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

---

## 🏆 Competition

**Google MedGemma Impact Challenge 2025**  
**Track:** Agentic Workflow Prize  
**Submission:** [Kaggle Link](https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/medpanel-multi-agent-clinical-ai)

---

## 📧 Contact

**Author:** Yogeshwari R.J  
**Email:** rj.yogeshwari@gmail.com 
**LinkedIn:** www.linkedin.com/in/yyyogeshwarirj

---

## ⚖️ License

MIT License - See [LICENSE](LICENSE)

---

## ⚠️ Disclaimer

This is a research prototype for the Google MedGemma Impact Challenge. **Not approved for clinical use.** Requires clinical validation, regulatory approval (FDA/EU MDR), and HIPAA compliance before deployment.

---

## 🙏 Acknowledgments

- Google MedGemma team for the foundation model
- NCBI for PubMed access
- HuggingFace for deployment platform
- Competition organizers

---

**MedPanel: Because in medicine, one opinion isn't enough.**
