"""
Benchmark: Single MedGemma vs MedPanel
Compares both approaches on the same test cases
"""

import json
import time
from benchmark_cases import TEST_CASES

# Import your MedPanel functions
from medpanel import run_medpanel, call_medgemma

def single_medgemma_approach(notes, image=None):
    """Standard single-agent approach (baseline)"""
    prompt = f"""You are a medical AI assistant. Analyze this clinical case and provide your primary diagnosis.
Clinical case:
{notes}
Provide a clear, concise primary diagnosis."""
    
    result = call_medgemma(prompt, image, max_tokens=200)
    return result


def evaluate_diagnosis(predicted, ground_truth):
    """
    Check if ground truth diagnosis is mentioned in the prediction.
    Returns True if found, False otherwise.
    """
    pred_lower = predicted.lower()
    truth_lower = ground_truth.lower()
    
    # Direct match
    if truth_lower in pred_lower:
        return True
    
    # Check synonyms
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
    
    return False


def run_comparison():
    """
    Main function: Compare Single MedGemma vs MedPanel
    """
    
    results = []
    
    print("="*70)
    print(" BENCHMARK: Single MedGemma vs MedPanel ".center(70))
    print("="*70)
    print(f"\nRunning {len(TEST_CASES)} test cases...")
    print(f"This will take approximately {len(TEST_CASES) * 1} minutes.\n")
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f" Case {case['id']}/{len(TEST_CASES)} ".center(70))
        print(f"{'='*70}")
        print(f"Symptoms: {case['notes'][:80]}...")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Difficulty: {case['difficulty'].upper()}")
        print(f"{'='*70}")
        
        # TEST 1: Single MedGemma
        print("\n🔹 Testing Single MedGemma Approach...")
        start_time = time.time()
        
        try:
            single_result = single_medgemma_approach(case['notes'], case.get('image'))
            single_time = time.time() - start_time
            single_correct = evaluate_diagnosis(single_result, case['ground_truth'])
            single_error = None
        except Exception as e:
            single_result = f"ERROR: {str(e)}"
            single_time = time.time() - start_time
            single_correct = False
            single_error = str(e)
            print(f"   ⚠️  Error: {e}")
        
        print(f"   Result: {single_result[:120]}...")
        print(f"   Time: {single_time:.1f}s")
        print(f"   {'✅ CORRECT' if single_correct else '❌ INCORRECT/MISSED'}")
        
        # TEST 2: MedPanel
        print("\n🔹 Testing MedPanel Approach...")
        start_time = time.time()
        
        try:
            medpanel_full_result = run_medpanel(case.get('image'), case['notes'])
            medpanel_time = time.time() - start_time
            
            # Parse the report
            report = medpanel_full_result['final_report']
            
            # Handle raw_response format
            if isinstance(report, dict) and "raw_response" in report:
                try:
                    raw = report["raw_response"]
                    # Clean truncated JSON
                    if not raw.strip().endswith('}'):
                        last_complete = raw.rfind('",')
                        if last_complete > 0:
                            raw = raw[:last_complete+2] + '\n}'
                    report = json.loads(raw)
                except:
                    # Keep raw_response if parsing fails
                    pass
            
            # Extract key info
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
        
        print(f"   Primary Diagnosis: {primary_dx}")
        print(f"   Panel Agreement: {confidence}%")
        print(f"   Escalated to Human: {escalated}")
        print(f"   Time: {medpanel_time:.1f}s")
        print(f"   {'✅ CORRECT' if medpanel_correct else '❌ INCORRECT/MISSED'}")
        
        # Check if Devil's Advocate made the difference
        devils_helped = False
        if medpanel_correct and not single_correct:
            devils_helped = True
            print(f"\n   🎯 DEVIL'S ADVOCATE MADE THE DIFFERENCE!")
            print(f"      Single agent missed it, but MedPanel caught it!")
        
        # Store results
        results.append({
            'case_id': case['id'],
            'case_summary': case['notes'][:80],
            'ground_truth': case['ground_truth'],
            'difficulty': case['difficulty'],
            'single': {
                'correct': single_correct,
                'time': single_time,
                'response': single_result[:250],
                'error': single_error
            },
            'medpanel': {
                'correct': medpanel_correct,
                'diagnosis': str(primary_dx)[:250],
                'confidence': confidence,
                'escalated': escalated,
                'time': medpanel_time,
                'error': medpanel_error
            },
            'devils_advocate_helped': devils_helped
        })
        
        # Rate limiting (be nice to API)
        if i < len(TEST_CASES):
            print("\n⏸  Waiting 3 seconds before next case...")
            time.sleep(3)
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print(f"\n\n{'='*70}")
    print(" FINAL RESULTS ".center(70, '='))
    print(f"{'='*70}\n")
    
    # Calculate metrics
    single_correct_count = sum(1 for r in results if r['single']['correct'])
    medpanel_correct_count = sum(1 for r in results if r['medpanel']['correct'])
    devils_saves = sum(1 for r in results if r['devils_advocate_helped'])
    
    single_accuracy = (single_correct_count / len(results)) * 100
    medpanel_accuracy = (medpanel_correct_count / len(results)) * 100
    improvement = medpanel_accuracy - single_accuracy
    
    print(f"📊 OVERALL ACCURACY:")
    print(f"   Single MedGemma:  {single_accuracy:5.1f}% ({single_correct_count}/{len(results)} correct)")
    print(f"   MedPanel:         {medpanel_accuracy:5.1f}% ({medpanel_correct_count}/{len(results)} correct)")
    print(f"   Improvement:      +{improvement:.1f} percentage points")
    
    print(f"\n🎯 DEVIL'S ADVOCATE IMPACT:")
    print(f"   Cases where Devil's Advocate made the difference: {devils_saves}")
    print(f"   That's {(devils_saves/len(results))*100:.0f}% of all cases!")
    
    # Breakdown by difficulty
    print(f"\n{'='*70}")
    print(" BREAKDOWN BY DIFFICULTY ".center(70))
    print(f"{'='*70}\n")
    
    for diff in ['easy', 'medium', 'hard']:
        diff_cases = [r for r in results if r['difficulty'] == diff]
        if diff_cases:
            single_acc = (sum(1 for r in diff_cases if r['single']['correct']) / len(diff_cases)) * 100
            medpanel_acc = (sum(1 for r in diff_cases if r['medpanel']['correct']) / len(diff_cases)) * 100
            diff_improvement = medpanel_acc - single_acc
            
            print(f"{diff.upper()} cases ({len(diff_cases)} total):")
            print(f"   Single:      {single_acc:5.1f}%")
            print(f"   MedPanel:    {medpanel_acc:5.1f}%")
            print(f"   Improvement: +{diff_improvement:.1f} points")
            print()
    
    # Save detailed results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_cases': len(results),
                'single_accuracy': single_accuracy,
                'medpanel_accuracy': medpanel_accuracy,
                'improvement': improvement,
                'devils_advocate_saves': devils_saves
            },
            'by_difficulty': {
                'easy': {
                    'count': len([r for r in results if r['difficulty'] == 'easy']),
                    'single': (sum(1 for r in results if r['difficulty'] == 'easy' and r['single']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'easy']))) * 100,
                    'medpanel': (sum(1 for r in results if r['difficulty'] == 'easy' and r['medpanel']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'easy']))) * 100
                },
                'medium': {
                    'count': len([r for r in results if r['difficulty'] == 'medium']),
                    'single': (sum(1 for r in results if r['difficulty'] == 'medium' and r['single']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'medium']))) * 100,
                    'medpanel': (sum(1 for r in results if r['difficulty'] == 'medium' and r['medpanel']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'medium']))) * 100
                },
                'hard': {
                    'count': len([r for r in results if r['difficulty'] == 'hard']),
                    'single': (sum(1 for r in results if r['difficulty'] == 'hard' and r['single']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'hard']))) * 100,
                    'medpanel': (sum(1 for r in results if r['difficulty'] == 'hard' and r['medpanel']['correct']) / max(1, len([r for r in results if r['difficulty'] == 'hard']))) * 100
                }
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"{'='*70}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    print("\n🚀 Starting benchmark comparison...\n")
    results = run_comparison()
    print("\n✅ Benchmark complete!\n")
