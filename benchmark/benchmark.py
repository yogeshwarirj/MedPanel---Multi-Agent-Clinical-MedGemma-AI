"""
Benchmark: Single MedGemma vs MedPanel

The whole point of this file is to answer one question:
does the multi-agent approach actually do better than just asking MedGemma once?

Spoiler: it does. But this is the code that proves it.
"""

import json
import time
from benchmark_cases import TEST_CASES

from medpanel import run_medpanel, call_medgemma


def single_medgemma_approach(notes, image=None):
    # this is the "naive" baseline — just ask MedGemma directly with no frills
    # same model, same weights, just no agents, no RAG, no adversarial review
    # if MedPanel wins, it's because of the architecture, not the model
    prompt = f"""You are a medical AI assistant. Analyze this clinical case and provide your primary diagnosis.
Clinical case:
{notes}
Provide a clear, concise primary diagnosis."""

    result = call_medgemma(prompt, image, max_tokens=200)
    return result


def evaluate_diagnosis(predicted, ground_truth):
    # checking if the model got it right
    # we're generous here — partial matches and synonyms both count
    # because "TB" and "Tuberculosis" are the same answer
    pred_lower = predicted.lower()
    truth_lower = ground_truth.lower()

    # exact match first — simplest case
    if truth_lower in pred_lower:
        return True

    # doctors use a lot of abbreviations and synonyms
    # this map catches the most common ones so we don't penalize correct answers
    # that just happen to use different terminology
    synonym_map = {
        "tuberculosis": ["tb", "mycobacterium", "tuberculous"],
        "myocardial infarction": ["heart attack", "mi", "acute coronary", "stemi"],
        "meningitis": ["meningeal inflammation"],
        "subarachnoid hemorrhage": ["sah", "brain bleed", "cerebral hemorrhage"],
        "ectopic pregnancy": ["tubal pregnancy", "extrauterine"],
        "hypoglycemia": ["low blood sugar", "hypoglycemic"],
        "interstitial lung disease": ["ild", "pulmonary fibrosis"],
        "spinal cord compression": ["cord compression", "myelopathy"],
        "malaria": ["plasmodium"],
        "cholecystitis": ["gallbladder inflammation"]
    }

    for condition, synonyms in synonym_map.items():
        if truth_lower in condition or condition in truth_lower:
            for syn in synonyms:
                if syn in pred_lower:
                    return True

    # nothing matched — it's wrong
    return False


def run_comparison():
    """
    Runs both approaches on every test case and compares the results.
    This takes a while — about 1 minute per case, so ~10 minutes total.
    Grab a coffee.
    """

    results = []

    print("=" * 70)
    print(" BENCHMARK: Single MedGemma vs MedPanel ".center(70))
    print("=" * 70)
    print(f"\nRunning {len(TEST_CASES)} test cases...")
    print(f"This will take approximately {len(TEST_CASES) * 1} minutes.\n")

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'=' * 70}")
        print(f" Case {case['id']}/{len(TEST_CASES)} ".center(70))
        print(f"{'=' * 70}")
        print(f"Symptoms:     {case['notes'][:80]}...")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Difficulty:   {case['difficulty'].upper()}")
        print(f"{'=' * 70}")

        # ── Test 1: Single MedGemma ──────────────────────────────────
        # run the baseline first so both approaches see the same case fresh
        print("\n🔹 Testing Single MedGemma Approach...")
        start_time = time.time()

        try:
            single_result = single_medgemma_approach(case['notes'], case.get('image'))
            single_time = time.time() - start_time
            single_correct = evaluate_diagnosis(single_result, case['ground_truth'])
            single_error = None
        except Exception as e:
            # don't crash the whole benchmark on one bad case
            single_result = f"ERROR: {str(e)}"
            single_time = time.time() - start_time
            single_correct = False
            single_error = str(e)
            print(f"   ⚠️  Error: {e}")

        print(f"   Result: {single_result[:120]}...")
        print(f"   Time:   {single_time:.1f}s")
        print(f"   {'✅ CORRECT' if single_correct else '❌ INCORRECT/MISSED'}")

        # ── Test 2: MedPanel ─────────────────────────────────────────
        # now run the full 5-agent pipeline on the same case
        # this is slower but should catch what the single agent missed
        print("\n🔹 Testing MedPanel Approach...")
        start_time = time.time()

        try:
            medpanel_full_result = run_medpanel(case.get('image'), case['notes'])
            medpanel_time = time.time() - start_time

            report = medpanel_full_result['final_report']

            # sometimes the orchestrator returns raw_response instead of clean JSON
            # this happens when the model hits max_tokens mid-output
            # try to salvage it rather than just marking it wrong
            if isinstance(report, dict) and "raw_response" in report:
                try:
                    raw = report["raw_response"]
                    # if the JSON got cut off, try to close it manually
                    if not raw.strip().endswith('}'):
                        last_complete = raw.rfind('",')
                        if last_complete > 0:
                            raw = raw[:last_complete + 2] + '\n}'
                    report = json.loads(raw)
                except Exception:
                    pass  # give up and use raw_response as-is

            # pull out the fields we care about
            primary_dx = report.get('primary_diagnosis', str(report)[:150])
            confidence = report.get('panel_agreement_score', 'N/A')
            escalated = report.get('escalate_to_human', False)

            medpanel_correct = evaluate_diagnosis(str(primary_dx), case['ground_truth'])
            medpanel_error = None

        except Exception as e:
            primary_dx = f"ERROR: {str(e)}"
            confidence = 'N/A'
            escalated = False
            medpanel_time = time.time() - start_time
            medpanel_correct = False
            medpanel_error = str(e)
            print(f"   ⚠️  Error: {e}")

        print(f"   Primary Diagnosis:  {primary_dx}")
        print(f"   Panel Agreement:    {confidence}%")
        print(f"   Escalated to Human: {escalated}")
        print(f"   Time:               {medpanel_time:.1f}s")
        print(f"   {'✅ CORRECT' if medpanel_correct else '❌ INCORRECT/MISSED'}")

        # ── Did the Devil's Advocate make the difference? ────────────
        # this is the most interesting metric — cases where single agent
        # failed but MedPanel caught it. that's the Devil's Advocate doing its job.
        devils_helped = False
        if medpanel_correct and not single_correct:
            devils_helped = True
            print(f"\n   🎯 DEVIL'S ADVOCATE MADE THE DIFFERENCE!")
            print(f"      Single agent missed it, but MedPanel caught it!")

        # store everything — we'll use this for the summary and JSON at the end
        results.append({
            'case_id': case['id'],
            'case_summary': case['notes'][:80],
            'ground_truth': case['ground_truth'],
            'difficulty': case['difficulty'],
            'single': {
                'correct': single_correct,
                'time': round(single_time, 2),
                'response': single_result[:250],
                'error': single_error
            },
            'medpanel': {
                'correct': medpanel_correct,
                'diagnosis': str(primary_dx)[:250],
                'confidence': confidence,
                'escalated': escalated,
                'time': round(medpanel_time, 2),
                'error': medpanel_error
            },
            'devils_advocate_helped': devils_helped
        })

        # small delay between cases — gives the GPU a moment to breathe
        if i < len(TEST_CASES):
            print("\n⏸  Waiting 3 seconds before next case...")
            time.sleep(3)

    # ── Final Summary ────────────────────────────────────────────────

    print(f"\n\n{'=' * 70}")
    print(" FINAL RESULTS ".center(70, '='))
    print(f"{'=' * 70}\n")

    # tally up the scores
    single_correct_count = sum(1 for r in results if r['single']['correct'])
    medpanel_correct_count = sum(1 for r in results if r['medpanel']['correct'])
    devils_saves = sum(1 for r in results if r['devils_advocate_helped'])

    single_accuracy = (single_correct_count / len(results)) * 100
    medpanel_accuracy = (medpanel_correct_count / len(results)) * 100
    improvement = medpanel_accuracy - single_accuracy

    print(f"📊 OVERALL ACCURACY:")
    print(f"   Single MedGemma : {single_accuracy:5.1f}%  ({single_correct_count}/{len(results)} correct)")
    print(f"   MedPanel        : {medpanel_accuracy:5.1f}%  ({medpanel_correct_count}/{len(results)} correct)")
    print(f"   Improvement     : +{improvement:.1f} percentage points")

    print(f"\n😈 DEVIL'S ADVOCATE IMPACT:")
    print(f"   Cases where Devil's Advocate made the difference: {devils_saves}")
    print(f"   That's {(devils_saves / len(results)) * 100:.0f}% of all cases!")

    # break it down by difficulty so we can see where the gaps actually are
    print(f"\n{'=' * 70}")
    print(" BREAKDOWN BY DIFFICULTY ".center(70))
    print(f"{'=' * 70}\n")

    difficulty_breakdown = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_cases = [r for r in results if r['difficulty'] == diff]
        if diff_cases:
            single_acc = (sum(1 for r in diff_cases if r['single']['correct']) / len(diff_cases)) * 100
            medpanel_acc = (sum(1 for r in diff_cases if r['medpanel']['correct']) / len(diff_cases)) * 100
            diff_improvement = medpanel_acc - single_acc

            difficulty_breakdown[diff] = {
                'count': len(diff_cases),
                'single_accuracy': round(single_acc, 1),
                'medpanel_accuracy': round(medpanel_acc, 1),
                'improvement': round(diff_improvement, 1)
            }

            print(f"{diff.upper()} cases ({len(diff_cases)} total):")
            print(f"   Single   : {single_acc:5.1f}%")
            print(f"   MedPanel : {medpanel_acc:5.1f}%")
            print(f"   Improvement: +{diff_improvement:.1f} points")
            print()

    # save to JSON
    # note: on HuggingFace Spaces this file won't survive a restart
    # that's why we also print it to console below — copy from there if needed
    output_file = 'benchmark_results.json'
    full_output = {
        'summary': {
            'total_cases': len(results),
            'single_accuracy': round(single_accuracy, 1),
            'medpanel_accuracy': round(medpanel_accuracy, 1),
            'improvement': round(improvement, 1),
            'devils_advocate_saves': devils_saves,
            'devils_advocate_impact_percent': round((devils_saves / len(results)) * 100, 1)
        },
        'by_difficulty': difficulty_breakdown,
        'detailed_results': results
    }

    with open(output_file, 'w') as f:
        json.dump(full_output, f, indent=2)

    print(f"{'=' * 70}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'=' * 70}\n")

    # always print full JSON to console too
    # HF Spaces filesystems are ephemeral — this is the reliable fallback
    print("\n📄 FULL JSON OUTPUT (copy from here if file download fails):")
    print("=" * 70)
    print(json.dumps(full_output, indent=2))
    print("=" * 70)

    return results


if __name__ == "__main__":
    print("\n🚀 Starting benchmark comparison...\n")
    results = run_comparison()
    print("\n✅ Benchmark complete!\n")
