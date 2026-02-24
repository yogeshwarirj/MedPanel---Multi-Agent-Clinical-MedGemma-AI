import gradio as gr
import subprocess
import json
import os

def run_benchmark():
    # spin up benchmark.py as a subprocess so it doesn't block the UI
    # -u flag = unbuffered output, which is what lets us stream line by line
    process = subprocess.Popen(
        ["python", "-u", "benchmark.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # hide model loading noise — it's distracting
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    # stream each line to the UI as it comes in
    # this way the user sees progress instead of staring at a blank box for 10 minutes
    output = ""
    for line in process.stdout:
        output += line
        yield output

    process.wait()

    # once benchmark.py finishes, try to load the saved JSON and show a clean summary
    # benchmark.py writes this file — if it's missing, something went wrong upstream
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
    # just a convenience button — lets you reload the last run without
    # running the full benchmark again (which takes ~10 minutes)
    if not os.path.exists("benchmark_results.json"):
        return "No results yet. Run the benchmark first."

    with open("benchmark_results.json", "r") as f:
        return json.dumps(json.load(f), indent=2)


# ── UI ───────────────────────────────────────────────────────────────

with gr.Blocks(title="MedPanel Benchmark") as demo:
    gr.Markdown("# 🏥 MedPanel Benchmark Runner")
    gr.Markdown("Compares **Single MedGemma** vs **MedPanel** across test cases.")

    with gr.Row():
        # run_benchmark streams live output as the test cases complete
        run_btn = gr.Button("▶️ Run Benchmark", variant="primary")
        # load_results just reads the last saved JSON — no rerun needed
        results_btn = gr.Button("📄 Load Saved Results")

    # tall textbox so you can see the full output without scrolling too much
    output_box = gr.Textbox(
        label="Benchmark Output",
        lines=40,
        max_lines=80,
    )

    run_btn.click(fn=run_benchmark, outputs=output_box)
    results_btn.click(fn=load_results, outputs=output_box)


demo.launch()
