import json
import pandas as pd
import os
import re
import glob

# === Configuration ===
PATCHED_DIR = "./data/"  # Directory where patched files are stored
OUTPUT_INDIVIDUAL_DIR = "./user_reports/"  # Directory to store individual reports
os.makedirs(OUTPUT_INDIVIDUAL_DIR, exist_ok=True)

def analyze_user(user_id):
    """
    For a specific user_id, read four files (same/same_o/other/other_o)
    and compute statistics
    """
    files = {
        'after_same': f"news120-sameother-edited/eval_test_same_results_{user_id}_patched.json",
        'before_same': f"news120-sameother-preedited/eval_test_same_resultso_{user_id}_patched.json",
        'after_other': f"news120-sameother-edited/eval_test_other_results_{user_id}_patched.json",
        'before_other': f"news120-sameother-preedited/eval_test_other_resultso_{user_id}_patched.json"
    }
    
    all_data = []
    
    for key, filename in files.items():
        path = os.path.join(PATCHED_DIR, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            
            is_before = 'before' in key
            
            for r in results:
                # Determine correctness
                ans = r.get('model_answer') or r.get('model_response', "")
                is_corr = str(r['target']).strip().lower() in str(ans).strip().lower()
                
                # Handle topic_cg (may be a list or a string)
                cgs = r.get('topic_cg', ['Unknown'])
                if isinstance(cgs, str): 
                    cgs = [cgs]
                
                for cg in cgs:
                    all_data.append({
                        'topic_cg': cg,
                        'is_before': is_before,
                        'is_correct': is_corr
                    })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Compute category statistics separately for Before and After
    def get_stats(sub_df):
        stats = sub_df.groupby('topic_cg').agg(
            Cases=('is_correct', 'count'),
            Correct=('is_correct', 'sum')
        ).reset_index()
        stats['Accuracy'] = (stats['Correct'] / stats['Cases'] * 100).round(2)
        return stats

    before_stats = get_stats(df[df['is_before'] == True])
    after_stats = get_stats(df[df['is_before'] == False])

    # Merge into a single table (similar to your desired output format)
    # Rename columns to distinguish Before/After
    before_stats = before_stats.rename(columns={'Correct': 'Num_Before', 'Accuracy': 'Acc_Before'})
    after_stats = after_stats.rename(columns={'Correct': 'Num_After', 'Accuracy': 'Acc_After'})

    # Merge based on topic_cg
    report = pd.merge(before_stats, after_stats, on=['topic_cg', 'Cases'], how='outer').fillna(0)
    
    # Calculate improvement (Down/Up)
    report['Down/Up'] = (report['Acc_After'] - report['Acc_Before']).round(2)
    
    # Compute Overall statistics
    overall = pd.DataFrame({
        'topic_cg': ['Overall'],
        'Cases': [report['Cases'].sum()],
        'Num_Before': [report['Num_Before'].sum()],
        'Acc_Before': [(report['Num_Before'].sum() / report['Cases'].sum() * 100).round(2)],
        'Num_After': [report['Num_After'].sum()],
        'Acc_After': [(report['Num_After'].sum() / report['Cases'].sum() * 100).round(2)],
        'Down/Up': [((report['Num_After'].sum() / report['Cases'].sum() * 100) - (report['Num_Before'].sum() / report['Cases'].sum() * 100)).round(2)]
    })

    final_report = pd.concat([overall, report.sort_values(by='Cases', ascending=False)], ignore_index=True)
    return final_report

# === Main Execution ===
# Extract all subject IDs (6-digit numbers from filenames)
user_ids = set()

all_files = glob.glob(os.path.join(PATCHED_DIR, "**/*.json"), recursive=True)
for f in all_files:
    match = re.search(r'\d{6}', os.path.basename(f))
    if match:
        user_ids.add(match.group(0))

print(f"Number of detected subjects: {len(user_ids)}")

for uid in sorted(user_ids):
    print(f"Generating report for subject {uid}...")
    report = analyze_user(uid)
    if report is not None:
        report.to_csv(f"{OUTPUT_INDIVIDUAL_DIR}/report_{uid}.csv", index=False, encoding='utf-8-sig')

print(f"\nAll individual reports have been saved to: {OUTPUT_INDIVIDUAL_DIR}")