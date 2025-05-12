import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import stats

# Load the dataset
df = pd.read_csv('all_bets.csv')

print(f"Loaded {len(df)} rows from all_bets.csv")

# First, let's check what experiment types are available and their counts
exp_type_counts = df['experiment_type'].value_counts()
print("\nExperiment type counts in original data:")
for exp_type, count in exp_type_counts.items():
   print(f"  {exp_type}: {count} rows")

# Reshape the data so each row is a single debate instance per model regardless of side
debate_data = []

# Group by debate_id, betting_model, and side to get all speech types for each debate-model-side combination
for (debate_id, model, side), group in df.groupby(['debate_id', 'betting_model', 'side']):
   # Extract data for each speech type if available
   opening = group[group['speech_type'] == 'opening']['bet_amount'].values
   rebuttal = group[group['speech_type'] == 'rebuttal']['bet_amount'].values
   closing = group[group['speech_type'] == 'closing']['bet_amount'].values

   # Check if we have all three speech types
   if len(opening) > 0 and len(rebuttal) > 0 and len(closing) > 0:
       exp_type = group['experiment_type'].iloc[0]
       topic = group['topic'].iloc[0]

       # For each individual complete set of speeches
       for i in range(min(len(opening), len(rebuttal), len(closing))):
           debate_data.append({
               'debate_id': debate_id,
               'betting_model': model,
               'experiment_type': exp_type,
               'side': side,
               'topic': topic,
               'opening_bet': opening[i] if i < len(opening) else np.nan,
               'rebuttal_bet': rebuttal[i] if i < len(rebuttal) else np.nan,
               'closing_bet': closing[i] if i < len(closing) else np.nan
           })

# Create a DataFrame with one row per debate-model-side instance
pivot_df = pd.DataFrame(debate_data)

# Check counts by experiment type in the pivoted data
pivot_exp_counts = pivot_df['experiment_type'].value_counts()
print("\nExperiment type counts in pivoted data:")
for exp_type, count in pivot_exp_counts.items():
   print(f"  {exp_type}: {count} rows")

# Get unique models and experiment types
models = sorted(pivot_df['betting_model'].unique())
experiment_types = sorted(pivot_df['experiment_type'].unique())

print(f"\nFound {len(models)} models and {len(experiment_types)} experiment types in the pivoted data")

# Function to format mean with standard deviation and sample size
def format_stat(values):
   values = [v for v in values if not np.isnan(v)]  # Filter out NaN values
   if len(values) == 0:
       return "N/A"
   mean = np.mean(values)
   std = np.std(values)
   count = len(values)
   return f"{mean:.2f} (±{std:.2f}, N={count})"

# Function to perform paired statistical tests and return formatted results
def paired_test(earlier_values, later_values, alpha=0.05):
   # Create pairs without NaN values
   pairs = [(e, l) for e, l in zip(earlier_values, later_values) if not np.isnan(e) and not np.isnan(l)]
   if not pairs:
       return "N/A"

   earlier = [p[0] for p in pairs]
   later = [p[1] for p in pairs]

   if len(earlier) != len(later) or len(earlier) == 0:
       return "N/A"

   # Check if data is paired correctly
   if len(earlier) < 2:
       return f"N={len(earlier)}, insufficient data"

   # Calculate mean difference
   mean_diff = np.mean(np.array(later) - np.array(earlier))

   # Perform paired t-test (one-sided, testing if later > earlier)
   t_stat, p_value = stats.ttest_rel(earlier, later)

   # Convert to one-sided p-value if mean difference is positive
   if mean_diff > 0:
       p_value = p_value / 2
   else:
       # If mean difference is negative, one-sided p-value is 1 - p_value/2
       p_value = 1 - (p_value / 2)

   # Determine significance
   significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < alpha else ""

   return f"Δ={mean_diff:.2f}, p={p_value:.4f}{significance}"

# For each experiment type, create a table showing all models and statistical tests
for exp_type in experiment_types:
   print(f"\n\n=== EXPERIMENT TYPE: {exp_type} ===")

   # Prepare data for tabulate
   table_data = []

   # For collecting overall paired data
   all_opening_bets = []
   all_rebuttal_bets = []
   all_closing_bets = []

   for model in models:
       # Filter data for this model and experiment type
       model_exp_data = pivot_df[(pivot_df['betting_model'] == model) & (pivot_df['experiment_type'] == exp_type)]

       if not model_exp_data.empty:
           # Get all bet values for each speech type
           opening_bets = model_exp_data['opening_bet'].tolist()
           rebuttal_bets = model_exp_data['rebuttal_bet'].tolist()
           closing_bets = model_exp_data['closing_bet'].tolist()

           # Add to overall stats
           all_opening_bets.extend(opening_bets)
           all_rebuttal_bets.extend(rebuttal_bets)
           all_closing_bets.extend(closing_bets)

           # Format descriptive stats
           opening_stats = format_stat(opening_bets)
           rebuttal_stats = format_stat(rebuttal_bets)
           closing_stats = format_stat(closing_bets)

           # Perform paired tests
           open_vs_rebuttal = paired_test(opening_bets, rebuttal_bets)
           rebuttal_vs_closing = paired_test(rebuttal_bets, closing_bets)
           open_vs_closing = paired_test(opening_bets, closing_bets)

           # Add row to table
           table_data.append([
               model,
               opening_stats,
               rebuttal_stats,
               closing_stats,
               open_vs_rebuttal,
               rebuttal_vs_closing,
               open_vs_closing
           ])
       else:
           # No data for this model in this experiment type
           table_data.append([model, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

   # Add overall row with paired tests across all models for this experiment type
   overall_opening_stats = format_stat(all_opening_bets)
   overall_rebuttal_stats = format_stat(all_rebuttal_bets)
   overall_closing_stats = format_stat(all_closing_bets)

   # Perform overall paired tests
   overall_open_vs_rebuttal = paired_test(all_opening_bets, all_rebuttal_bets)
   overall_rebuttal_vs_closing = paired_test(all_rebuttal_bets, all_closing_bets)
   overall_open_vs_closing = paired_test(all_opening_bets, all_closing_bets)

   table_data.append([
       "OVERALL",
       overall_opening_stats,
       overall_rebuttal_stats,
       overall_closing_stats,
       overall_open_vs_rebuttal,
       overall_rebuttal_vs_closing,
       overall_open_vs_closing
   ])

   # Print table with headers
   headers = [
       "Model",
       "Opening Bet",
       "Rebuttal Bet",
       "Closing Bet",
       "Open→Rebuttal",
       "Rebuttal→Closing",
       "Open→Closing"
   ]
   print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Create a summary table by model (across all experiment types)
print("\n\n=== OVERALL MODEL AVERAGES & ESCALATION (ACROSS ALL EXPERIMENT TYPES) ===")
table_data = []

# For collecting grand overall stats
all_opening_bets = []
all_rebuttal_bets = []
all_closing_bets = []

for model in models:
   # Filter data for this model across all experiment types
   model_data = pivot_df[pivot_df['betting_model'] == model]

   if not model_data.empty:
       # Get all bet values for each speech type
       opening_bets = model_data['opening_bet'].tolist()
       rebuttal_bets = model_data['rebuttal_bet'].tolist()
       closing_bets = model_data['closing_bet'].tolist()

       # Add to overall stats
       all_opening_bets.extend(opening_bets)
       all_rebuttal_bets.extend(rebuttal_bets)
       all_closing_bets.extend(closing_bets)

       # Format descriptive stats
       opening_stats = format_stat(opening_bets)
       rebuttal_stats = format_stat(rebuttal_bets)
       closing_stats = format_stat(closing_bets)

       # Perform paired tests
       open_vs_rebuttal = paired_test(opening_bets, rebuttal_bets)
       rebuttal_vs_closing = paired_test(rebuttal_bets, closing_bets)
       open_vs_closing = paired_test(opening_bets, closing_bets)

       # Add row to table
       table_data.append([
           model,
           opening_stats,
           rebuttal_stats,
           closing_stats,
           open_vs_rebuttal,
           rebuttal_vs_closing,
           open_vs_closing
       ])
   else:
       table_data.append([model, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

# Add grand overall row
overall_opening_stats = format_stat(all_opening_bets)
overall_rebuttal_stats = format_stat(all_rebuttal_bets)
overall_closing_stats = format_stat(all_closing_bets)

# Overall paired tests
grand_open_vs_rebuttal = paired_test(all_opening_bets, all_rebuttal_bets)
grand_rebuttal_vs_closing = paired_test(all_rebuttal_bets, all_closing_bets)
grand_open_vs_closing = paired_test(all_opening_bets, all_closing_bets)

table_data.append([
   "GRAND OVERALL",
   overall_opening_stats,
   overall_rebuttal_stats,
   overall_closing_stats,
   grand_open_vs_rebuttal,
   grand_rebuttal_vs_closing,
   grand_open_vs_closing
])

headers = [
   "Model",
   "Opening Bet",
   "Rebuttal Bet",
   "Closing Bet",
   "Open→Rebuttal",
   "Rebuttal→Closing",
   "Open→Closing"
]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Count models with significant escalation per experiment type
print("\n\n=== SUMMARY OF MODELS WITH SIGNIFICANT ESCALATION ===")
summary_data = []

for exp_type in experiment_types:
   # Initialize counters
   sig_open_to_rebuttal = 0
   sig_rebuttal_to_closing = 0
   sig_open_to_closing = 0
   total_models_with_data = 0

   for model in models:
       model_exp_data = pivot_df[(pivot_df['betting_model'] == model) & (pivot_df['experiment_type'] == exp_type)]

       if not model_exp_data.empty:
           # Get bet values
           opening_bets = model_exp_data['opening_bet'].tolist()
           rebuttal_bets = model_exp_data['rebuttal_bet'].tolist()
           closing_bets = model_exp_data['closing_bet'].tolist()

           # Only count if we have paired data
           pairs_or_rebuttal = [(o, r) for o, r in zip(opening_bets, rebuttal_bets) if not np.isnan(o) and not np.isnan(r)]
           pairs_rc = [(r, c) for r, c in zip(rebuttal_bets, closing_bets) if not np.isnan(r) and not np.isnan(c)]
           pairs_oc = [(o, c) for o, c in zip(opening_bets, closing_bets) if not np.isnan(o) and not np.isnan(c)]

           if len(pairs_or_rebuttal) >= 2 or len(pairs_rc) >= 2 or len(pairs_oc) >= 2:
               total_models_with_data += 1

               # Check for significance in each transition
               if paired_test(opening_bets, rebuttal_bets).find("*") != -1:
                   sig_open_to_rebuttal += 1

               if paired_test(rebuttal_bets, closing_bets).find("*") != -1:
                   sig_rebuttal_to_closing += 1

               if paired_test(opening_bets, closing_bets).find("*") != -1:
                   sig_open_to_closing += 1

   # Add summary row
   summary_data.append([
       exp_type,
       f"{sig_open_to_rebuttal}/{total_models_with_data}" if total_models_with_data > 0 else "N/A",
       f"{sig_rebuttal_to_closing}/{total_models_with_data}" if total_models_with_data > 0 else "N/A",
       f"{sig_open_to_closing}/{total_models_with_data}" if total_models_with_data > 0 else "N/A"
   ])

# Print summary table
summary_headers = [
   "Experiment Type",
   "Models with Sig. Open→Rebuttal",
   "Models with Sig. Rebuttal→Closing",
   "Models with Sig. Open→Closing"
]
print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))
