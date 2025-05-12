import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MultipleLocator

# Load the dataset
df = pd.read_csv('all_bets.csv')

# Reshape the data similar to previous script
debate_data = []
for (debate_id, model, side), group in df.groupby(['debate_id', 'betting_model', 'side']):
    opening = group[group['speech_type'] == 'opening']['bet_amount'].values
    rebuttal = group[group['speech_type'] == 'rebuttal']['bet_amount'].values
    closing = group[group['speech_type'] == 'closing']['bet_amount'].values

    if len(opening) > 0 and len(rebuttal) > 0 and len(closing) > 0:
        exp_type = group['experiment_type'].iloc[0]

        for i in range(min(len(opening), len(rebuttal), len(closing))):
            debate_data.append({
                'debate_id': debate_id,
                'betting_model': model,
                'experiment_type': exp_type,
                'side': side,
                'opening_bet': opening[i] if i < len(opening) else np.nan,
                'rebuttal_bet': rebuttal[i] if i < len(rebuttal) else np.nan,
                'closing_bet': closing[i] if i < len(closing) else np.nan
            })

pivot_df = pd.DataFrame(debate_data)

# Create a mapping for the experiment type names
exp_type_map = {
    'cross_model': 'Cross-model',
    'informed_self': 'Informed Self',
    'public_bets': 'Public Bets',
    'self_debate': 'Standard Self'
}

# Apply the mapping to the experiment_type column
pivot_df['experiment_display_name'] = pivot_df['experiment_type'].map(exp_type_map)

# Prepare data for plotting
# Calculate mean and standard error for each experiment type and speech stage
summary_data = []

for exp_type in pivot_df['experiment_type'].unique():
    exp_data = pivot_df[pivot_df['experiment_type'] == exp_type]
    display_name = exp_type_map.get(exp_type, exp_type)

    # Calculate means
    opening_mean = np.nanmean(exp_data['opening_bet'])
    rebuttal_mean = np.nanmean(exp_data['rebuttal_bet'])
    closing_mean = np.nanmean(exp_data['closing_bet'])

    # Calculate standard errors
    opening_se = np.nanstd(exp_data['opening_bet']) / np.sqrt(np.sum(~np.isnan(exp_data['opening_bet'])))
    rebuttal_se = np.nanstd(exp_data['rebuttal_bet']) / np.sqrt(np.sum(~np.isnan(exp_data['rebuttal_bet'])))
    closing_se = np.nanstd(exp_data['closing_bet']) / np.sqrt(np.sum(~np.isnan(exp_data['closing_bet'])))

    # Perform paired t-tests for significance (keeping calculation but not showing stars)
    # Opening -> Rebuttal
    open_rebuttal_pairs = [(o, r) for o, r in zip(exp_data['opening_bet'], exp_data['rebuttal_bet'])
                          if not np.isnan(o) and not np.isnan(r)]
    if len(open_rebuttal_pairs) >= 2:
        o_vals, r_vals = zip(*open_rebuttal_pairs)
        t_stat_or, p_value_or = stats.ttest_rel(o_vals, r_vals)
        # One-sided test
        if np.mean(r_vals) > np.mean(o_vals):
            p_value_or = p_value_or / 2
        else:
            p_value_or = 1 - (p_value_or / 2)
    else:
        p_value_or = 1.0

    # Rebuttal -> Closing
    rebuttal_closing_pairs = [(r, c) for r, c in zip(exp_data['rebuttal_bet'], exp_data['closing_bet'])
                              if not np.isnan(r) and not np.isnan(c)]
    if len(rebuttal_closing_pairs) >= 2:
        r_vals, c_vals = zip(*rebuttal_closing_pairs)
        t_stat_rc, p_value_rc = stats.ttest_rel(r_vals, c_vals)
        # One-sided test
        if np.mean(c_vals) > np.mean(r_vals):
            p_value_rc = p_value_rc / 2
        else:
            p_value_rc = 1 - (p_value_rc / 2)
    else:
        p_value_rc = 1.0

    summary_data.append({
        'experiment_type': exp_type,
        'display_name': display_name,
        'opening_mean': opening_mean,
        'rebuttal_mean': rebuttal_mean,
        'closing_mean': closing_mean,
        'opening_se': opening_se,
        'rebuttal_se': rebuttal_se,
        'closing_se': closing_se,
        'p_value_or': p_value_or,
        'p_value_rc': p_value_rc
    })

# Also calculate overall average across all experiment types
all_data = pivot_df
opening_mean = np.nanmean(all_data['opening_bet'])
rebuttal_mean = np.nanmean(all_data['rebuttal_bet'])
closing_mean = np.nanmean(all_data['closing_bet'])

opening_se = np.nanstd(all_data['opening_bet']) / np.sqrt(np.sum(~np.isnan(all_data['opening_bet'])))
rebuttal_se = np.nanstd(all_data['rebuttal_bet']) / np.sqrt(np.sum(~np.isnan(all_data['rebuttal_bet'])))
closing_se = np.nanstd(all_data['closing_bet']) / np.sqrt(np.sum(~np.isnan(all_data['closing_bet'])))

# Overall significance tests
open_rebuttal_pairs = [(o, r) for o, r in zip(all_data['opening_bet'], all_data['rebuttal_bet'])
                      if not np.isnan(o) and not np.isnan(r)]
if len(open_rebuttal_pairs) >= 2:
    o_vals, r_vals = zip(*open_rebuttal_pairs)
    t_stat_or, p_value_or = stats.ttest_rel(o_vals, r_vals)
    # One-sided test
    if np.mean(r_vals) > np.mean(o_vals):
        p_value_or = p_value_or / 2
    else:
        p_value_or = 1 - (p_value_or / 2)
else:
    p_value_or = 1.0

rebuttal_closing_pairs = [(r, c) for r, c in zip(all_data['rebuttal_bet'], all_data['closing_bet'])
                          if not np.isnan(r) and not np.isnan(c)]
if len(rebuttal_closing_pairs) >= 2:
    r_vals, c_vals = zip(*rebuttal_closing_pairs)
    t_stat_rc, p_value_rc = stats.ttest_rel(r_vals, c_vals)
    # One-sided test
    if np.mean(c_vals) > np.mean(r_vals):
        p_value_rc = p_value_rc / 2
    else:
        p_value_rc = 1 - (p_value_rc / 2)
else:
    p_value_rc = 1.0

summary_data.append({
    'experiment_type': 'Overall',
    'display_name': 'Overall',
    'opening_mean': opening_mean,
    'rebuttal_mean': rebuttal_mean,
    'closing_mean': closing_mean,
    'opening_se': opening_se,
    'rebuttal_se': rebuttal_se,
    'closing_se': closing_se,
    'p_value_or': p_value_or,
    'p_value_rc': p_value_rc
})

# Create DataFrame with summary data
summary_df = pd.DataFrame(summary_data)

# Set a clean style with good readability
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)

# Create the figure with properly sized subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each experiment type
colors = {
    'cross_model': '#1f77b4',     # blue
    'self_debate': '#ff7f0e',     # orange
    'informed_self': '#2ca02c',   # green
    'public_bets': '#d62728',     # red
    'Overall': '#7f7f7f'          # gray
}

# Plot each experiment type
x_positions = [0, 1, 2]  # Opening, Rebuttal, Closing

for i, row in summary_df.iterrows():
    exp_type = row['experiment_type']
    display_name = row['display_name']

    # Get the means and standard errors
    means = [row['opening_mean'], row['rebuttal_mean'], row['closing_mean']]
    errors = [row['opening_se'], row['rebuttal_se'], row['closing_se']]

    # Plot the line
    if exp_type == 'Overall':
        ax.plot(x_positions, means, label=display_name, color=colors[exp_type],
                linestyle='--', linewidth=2.5, marker='o', markersize=8)
    else:
        ax.plot(x_positions, means, label=display_name, color=colors[exp_type],
                linestyle='-', linewidth=2, marker='o', markersize=7)

    # Add error bars
    ax.errorbar(x_positions, means, yerr=errors, fmt='none', color=colors[exp_type],
                capsize=5, capthick=1, elinewidth=1)

# Set labels and title
ax.set_xlabel('Debate Stage', fontsize=14)
ax.set_ylabel('Confidence Score (0-100)', fontsize=14)
ax.set_title('Confidence Escalation Across Debate Stages', fontsize=16, pad=20)

# Set the x-tick labels
ax.set_xticks(x_positions)
ax.set_xticklabels(['Opening', 'Rebuttal', 'Closing'], fontsize=12)

# Set the y-axis to start at a reasonable value (maybe 40 or 50)
ymin = max(40, np.floor(min([row['opening_mean'] for _, row in summary_df.iterrows()]) - 5))
ax.set_ylim(ymin, 100)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

# Add a horizontal line at 50% (fair confidence)
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.7)
ax.text(2.1, 50, 'Fair confidence (50%)', va='center', color='gray', fontsize=10)

# Add legend
ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=11)

# Adjust layout
plt.tight_layout()

# Save to PDF for LaTeX
plt.savefig('confidence_escalation.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Also save as PNG for easy viewing
plt.savefig('confidence_escalation.png', format='png', dpi=300, bbox_inches='tight')

print("Visualization saved as 'confidence_escalation.pdf' and 'confidence_escalation.png'")
