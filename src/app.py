# app.py
# Gradio interface for MedPanel on HuggingFace Spaces.
# This file is the entry point — HF Spaces looks for app.py automatically.
# It wraps the medpanel.py pipeline in a clean web UI.

import json
import gradio as gr
from PIL import Image

from medpanel import run_medpanel


# ── Helper: Format the Final Report for Display ──────────────────────

def format_report(report):
    # If empty or None
    if not report:
        return "⚠️ No report generated."

    # If it's a string already — just return it
    if isinstance(report, str):
        return report

    # ✅ FIX: raw_response branch — handle empty string case
    if "raw_response" in report:
        raw = report['raw_response']
        if not raw or not raw.strip():
            return "⚠️ Model returned an empty response. Please try running again."
        return f"📋 REPORT\n\n{raw}"

    # Normal structured report
    lines = []

    primary = report.get("primary_diagnosis", "Unknown")
    lines.append(f"🔬 PRIMARY DIAGNOSIS\n{primary}\n")

    differentials = report.get("differential_diagnoses", [])
    if differentials:
        lines.append("📋 DIFFERENTIAL DIAGNOSES")
        for i, d in enumerate(differentials, 1):
            lines.append(f"   {i}. {d}")
        lines.append("")

    score = report.get("panel_agreement_score", "N/A")
    lines.append(f"🤝 PANEL AGREEMENT SCORE\n{score}/100\n")

    red_flags = report.get("red_flags", [])
    if red_flags:
        lines.append("🚨 RED FLAGS")
        for flag in red_flags:
            lines.append(f"   • {flag}")
        lines.append("")

    next_steps = report.get("recommended_next_steps", [])
    if next_steps:
        lines.append("📌 RECOMMENDED NEXT STEPS")
        for step in next_steps:
            lines.append(f"   • {step}")
        lines.append("")

    escalate = report.get("escalate_to_human", False)
    reason = report.get("escalation_reason", "Not required")
    icon = "🔴" if escalate else "🟢"
    lines.append(f"{icon} ESCALATION TO HUMAN DOCTOR")
    lines.append(f"   {'REQUIRED' if escalate else 'Not required'}: {reason}\n")

    summary = report.get("patient_summary", "")
    if summary:
        lines.append(f"💬 PATIENT SUMMARY\n{summary}")

    result = "\n".join(lines)

    # ✅ FIX: if structured formatting produced nothing, fall back to raw JSON
    if not result.strip():
        return json.dumps(report, indent=2)

    return result


# ── Main Inference Function ───────────────────────────────────────────

def analyze(image, clinical_notes):
    if not clinical_notes or clinical_notes.strip() == "":
        return (
            "⚠️ Please enter clinical notes before submitting.",
            "No trace available."
        )

    pil_image = None
    if image is not None:
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image

    try:
        results = run_medpanel(pil_image, clinical_notes)
        final_report = results.get("final_report", {})
        trace = results.get("panel_trace", [])

        report_text = format_report(final_report)
        trace_text = json.dumps(trace, indent=2, default=str)

        # Final safety fallback — if report_text is still empty
        if not report_text or report_text.strip() == "":
            report_text = json.dumps(final_report, indent=2, default=str)

        return report_text, trace_text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error: {str(e)}\n\n{error_details}", "Error occurred."


# ── Example Cases ────────────────────────────────────────────────────

examples = [
    [
        None,
        "65-year-old male. Persistent cough for 3 months, night sweats, weight loss of 8kg. "
        "Recent travel to high TB-prevalence region. Low-grade fever. No prior TB history."
    ],
    [
        None,
        "45-year-old female. Sudden chest pain radiating to left arm, shortness of breath. "
        "History of hypertension and high cholesterol. Smoker for 20 years. ECG changes noted."
    ],
    [
        None,
        "32-year-old male. Severe headache, photophobia, neck stiffness, fever 39.5°C. "
        "Petechial rash on lower limbs. Symptoms started 12 hours ago."
    ]
]


# ── Build the Gradio Interface ────────────────────────────────────────

with gr.Blocks(
    title="MedPanel — Multi-Agent AI Diagnostic System",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    # Header
    gr.Markdown("""
    # 🏥 MedPanel
    ### Multi-Agent AI Clinical Decision Support System
    Built with **Google MedGemma** (HAI-DEF) | Four specialized agents + PubMed RAG
    > ⚠️ **Disclaimer:** This system is for research and demonstration purposes only.
    > It is not a substitute for professional medical advice, diagnosis, or treatment.
    """)

    with gr.Row():

        # Left column — inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")

            image_input = gr.Image(
                label="Medical Image (X-ray, CT scan — optional)",
                type="numpy"
            )

            notes_input = gr.Textbox(
                label="Clinical Notes",
                placeholder="Enter patient symptoms, history, vitals...\n\nExample: 65yo male, 3-month cough, night sweats, weight loss, recent travel to TB-endemic region.",
                lines=8
            )

            submit_btn = gr.Button(
                "🔬 Run MedPanel Analysis",
                variant="primary",
                size="lg"
            )

        # Right column — outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Results")

            with gr.Tabs():

                with gr.TabItem("📋 Final Report"):
                    report_output = gr.Textbox(
                        label="MedPanel Diagnostic Report",
                        lines=20,
                        interactive=False
                    )

                with gr.TabItem("🔍 Agent Trace"):
                    gr.Markdown("Raw output from each of the 4 agents — useful for understanding the reasoning.")
                    trace_output = gr.Textbox(
                        label="Panel Trace (JSON)",
                        lines=20,
                        interactive=False
                    )

    # Connect submit button
    submit_btn.click(
        fn=analyze,
        inputs=[image_input, notes_input],
        outputs=[report_output, trace_output]
    )

    # Example cases
    gr.Markdown("### 💡 Try These Example Cases")
    gr.Examples(
        examples=examples,
        inputs=[image_input, notes_input],
        label="Click any example to pre-fill the inputs"
    )

    # How it works
    gr.Markdown("""
    ---
    ### 🧠 How MedPanel Works
    | Agent | Role |
    |-------|------|
    | 🩻 Radiologist | Analyzes the medical image for visual findings |
    | 🩺 Internist | Reviews clinical notes and builds a differential |
    | 📚 Evidence Reviewer | Fetches relevant PubMed literature via RAG |
    | 😈 Devil's Advocate | Challenges findings and catches missed diagnoses |
    | 🎯 Orchestrator | Synthesizes all inputs into the final report |
    """)


# ── Launch ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue(
        max_size=5,
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
