import gradio as gr      
import subprocess
import json
import os

def run_benchmark():
    process = subprocess.Popen(
        ["python", "-u", "benchmark.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    output = ""
    for line in process.stdout:
        output += line
        yield output

    process.wait()

    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", "r") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        output += f"\n\n📊 SUMMARY:\n"
        output += f"  Single MedGemma Accuracy : {summary.get('single_accuracy', 'N/A'):.1f}%\n"
        output += f"  MedPanel Accuracy        : {summary.get('medpanel_accuracy', 'N/A'):.1f}%\n"
        output += f"  Improvement              : +{summary.get('improvement', 'N/A'):.1f} pts\n"
        output += f"  Devil's Advocate Saves   : {summary.get('devils_advocate_saves', 'N/A')}\n"
        yield output

def load_results():
    if not os.path.exists("benchmark_results.json"):
        return "No results yet. Run the benchmark first."
    with open("benchmark_results.json", "r") as f:
        return json.dumps(json.load(f), indent=2)

with gr.Blocks(title="MedPanel Benchmark") as demo:
    gr.Markdown("# 🏥 MedPanel Benchmark Runner")
    gr.Markdown("Compares **Single MedGemma** vs **MedPanel** across test cases.")

    with gr.Row():
        run_btn = gr.Button("▶️ Run Benchmark", variant="primary")
        results_btn = gr.Button("📄 Load Saved Results")

    output_box = gr.Textbox(
        label="Benchmark Output",
        lines=40,
        max_lines=80,
    )

    run_btn.click(fn=run_benchmark, outputs=output_box)
    results_btn.click(fn=load_results, outputs=output_box)

demo.launch()  
