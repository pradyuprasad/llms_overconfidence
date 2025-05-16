import json
import pandas as pd
from pathlib import Path

def analyze_assessment_by_numeric_confidence():
    # List of JSON files to analyze
    files = [
        "bet_analysis_results_private_self_bet.json",
        "bet_analysis_results_private_self_bet_anchored.json",
        "bet_analysis_results_public_bets.json",
        "bet_analysis_results_private_bet_experiments_diff_models.json"
    ]

    # Process each file separately
    for file_path in files:
        try:
            with open(Path(file_path), 'r') as f:
                data = json.load(f)
                print(f"\n\n=== ANALYSIS FOR {file_path} ===")
                print(f"Total entries: {len(data)}")

                # Extract relevant data into a dataframe
                rows = []
                for entry in data:
                    # Skip entries with errors
                    if 'analysis' not in entry or 'error' in entry['analysis']:
                        continue

                    # Extract alignment info
                    alignment = entry['analysis']['betting_alignment']

                    # Check if numeric_confidence_present exists, default to False if not
                    numeric_present = alignment.get('numeric_confidence_present', False)

                    row = {
                        'numeric_confidence_present': numeric_present,
                        'assessment': alignment['assessment'],
                        'degree': alignment['degree']
                    }
                    rows.append(row)

                # Create dataframe
                df = pd.DataFrame(rows)
                if len(df) == 0:
                    print("No valid data found in this file")
                    continue

                # Calculate overall distribution
                print("\nOVERALL ASSESSMENT DISTRIBUTION:")
                assessment_counts = df['assessment'].value_counts()
                assessment_pct = (assessment_counts / len(df) * 100).round(1)
                for assessment, count in assessment_counts.items():
                    pct = assessment_pct[assessment]
                    print(f"{assessment}: {count} ({pct}%)")

                print("\nOVERALL DEGREE DISTRIBUTION:")
                degree_counts = df['degree'].value_counts()
                degree_pct = (degree_counts / len(df) * 100).round(1)
                for degree, count in degree_counts.items():
                    pct = degree_pct[degree]
                    print(f"{degree}: {count} ({pct}%)")

                # Split by numeric confidence present
                print("\nASSESSMENT DISTRIBUTION (SPLIT BY NUMERIC CONFIDENCE PRESENT):")
                numeric_assessment = pd.crosstab(
                    df['numeric_confidence_present'],
                    df['assessment'],
                    normalize='index'
                ) * 100
                print(numeric_assessment.round(1))

                print("\nDEGREE DISTRIBUTION (SPLIT BY NUMERIC CONFIDENCE PRESENT):")
                numeric_degree = pd.crosstab(
                    df['numeric_confidence_present'],
                    df['degree'],
                    normalize='index'
                ) * 100
                print(numeric_degree.round(1))

                # Show count of numeric vs non-numeric
                print("\nNUMERIC CONFIDENCE PRESENT COUNTS:")
                numeric_counts = df['numeric_confidence_present'].value_counts()
                numeric_pct = (numeric_counts / len(df) * 100).round(1)
                for present, count in numeric_counts.items():
                    pct = numeric_pct[present]
                    print(f"{present}: {count} ({pct}%)")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    analyze_assessment_by_numeric_confidence()
