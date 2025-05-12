import pandas as pd

# Load the dataset
df = pd.read_csv('all_bets.csv')

# Filter to only include closing bets
closing_bets = df[df['speech_type'] == 'closing']

# Function to categorize a bet
def categorize_bet(bet):
    if bet <= 50:
        return "low"  # ≤50
    elif 51 <= bet <= 75:
        return "medium"  # 51-75
    else:
        return "high"  # >75

# Dictionary to store results
results = {}

# Analyze cross_model debates
cross_model_bets = closing_bets[closing_bets['experiment_type'] == 'cross_model']
cross_model_pairs = []

for debate_id in cross_model_bets['debate_id'].unique():
    debate_data = cross_model_bets[cross_model_bets['debate_id'] == debate_id]

    # Skip if we don't have exactly one proposition and one opposition bet
    if len(debate_data[debate_data['side'] == 'proposition']) != 1 or len(debate_data[debate_data['side'] == 'opposition']) != 1:
        continue

    prop_bet = debate_data[debate_data['side'] == 'proposition']['bet_amount'].iloc[0]
    opp_bet = debate_data[debate_data['side'] == 'opposition']['bet_amount'].iloc[0]

    cross_model_pairs.append((prop_bet, opp_bet))

# Count patterns for cross_model
cm_both_low = sum(1 for p, o in cross_model_pairs if p <= 50 and o <= 50)
cm_both_medium = sum(1 for p, o in cross_model_pairs if 51 <= p <= 75 and 51 <= o <= 75)
cm_both_high = sum(1 for p, o in cross_model_pairs if p > 75 and o > 75)
cm_low_medium = sum(1 for p, o in cross_model_pairs if (p <= 50 and 51 <= o <= 75) or (o <= 50 and 51 <= p <= 75))
cm_low_high = sum(1 for p, o in cross_model_pairs if (p <= 50 and o > 75) or (o <= 50 and p > 75))
cm_medium_high = sum(1 for p, o in cross_model_pairs if (51 <= p <= 75 and o > 75) or (51 <= o <= 75 and p > 75))

total_cm = len(cross_model_pairs)
results['cross_model'] = {
    'total': total_cm,
    'both_low': cm_both_low,
    'both_medium': cm_both_medium,
    'both_high': cm_both_high,
    'low_medium': cm_low_medium,
    'low_high': cm_low_high,
    'medium_high': cm_medium_high,
    'pct_both_low': (cm_both_low/total_cm*100) if total_cm > 0 else 0,
    'pct_both_medium': (cm_both_medium/total_cm*100) if total_cm > 0 else 0,
    'pct_both_high': (cm_both_high/total_cm*100) if total_cm > 0 else 0,
    'pct_low_medium': (cm_low_medium/total_cm*100) if total_cm > 0 else 0,
    'pct_low_high': (cm_low_high/total_cm*100) if total_cm > 0 else 0,
    'pct_medium_high': (cm_medium_high/total_cm*100) if total_cm > 0 else 0,
}

# Initialize variables to track overall counts
all_pairs = cross_model_pairs.copy()

# Analyze self-debate formats
for exp_type in ['self_debate', 'informed_self', 'public_bets']:
    exp_bets = closing_bets[closing_bets['experiment_type'] == exp_type]
    model_pairs = []

    for debate_id in exp_bets['debate_id'].unique():
        debate_data = exp_bets[exp_bets['debate_id'] == debate_id]

        # Get all models that appear on both sides
        for model in debate_data['betting_model'].unique():
            prop_bets = debate_data[(debate_data['side'] == 'proposition') &
                                   (debate_data['betting_model'] == model)]
            opp_bets = debate_data[(debate_data['side'] == 'opposition') &
                                  (debate_data['betting_model'] == model)]

            if not prop_bets.empty and not opp_bets.empty:
                # For each prop/opp combination of this model
                for _, prop_row in prop_bets.iterrows():
                    for _, opp_row in opp_bets.iterrows():
                        pair = (prop_row['bet_amount'], opp_row['bet_amount'])
                        model_pairs.append(pair)
                        all_pairs.append(pair)  # Add to overall total

    # Count patterns
    both_low = sum(1 for p, o in model_pairs if p <= 50 and o <= 50)
    both_medium = sum(1 for p, o in model_pairs if 51 <= p <= 75 and 51 <= o <= 75)
    both_high = sum(1 for p, o in model_pairs if p > 75 and o > 75)
    low_medium = sum(1 for p, o in model_pairs if (p <= 50 and 51 <= o <= 75) or (o <= 50 and 51 <= p <= 75))
    low_high = sum(1 for p, o in model_pairs if (p <= 50 and o > 75) or (o <= 50 and p > 75))
    medium_high = sum(1 for p, o in model_pairs if (51 <= p <= 75 and o > 75) or (51 <= o <= 75 and p > 75))

    total = len(model_pairs)
    results[exp_type] = {
        'total': total,
        'both_low': both_low,
        'both_medium': both_medium,
        'both_high': both_high,
        'low_medium': low_medium,
        'low_high': low_high,
        'medium_high': medium_high,
        'pct_both_low': (both_low/total*100) if total > 0 else 0,
        'pct_both_medium': (both_medium/total*100) if total > 0 else 0,
        'pct_both_high': (both_high/total*100) if total > 0 else 0,
        'pct_low_medium': (low_medium/total*100) if total > 0 else 0,
        'pct_low_high': (low_high/total*100) if total > 0 else 0,
        'pct_medium_high': (medium_high/total*100) if total > 0 else 0,
    }

# Calculate "overall" stats across all experiment types
total_all = len(all_pairs)
all_both_low = sum(1 for p, o in all_pairs if p <= 50 and o <= 50)
all_both_medium = sum(1 for p, o in all_pairs if 51 <= p <= 75 and 51 <= o <= 75)
all_both_high = sum(1 for p, o in all_pairs if p > 75 and o > 75)
all_low_medium = sum(1 for p, o in all_pairs if (p <= 50 and 51 <= o <= 75) or (o <= 50 and 51 <= p <= 75))
all_low_high = sum(1 for p, o in all_pairs if (p <= 50 and o > 75) or (o <= 50 and p > 75))
all_medium_high = sum(1 for p, o in all_pairs if (51 <= p <= 75 and o > 75) or (51 <= o <= 75 and p > 75))

results['overall'] = {
    'total': total_all,
    'both_low': all_both_low,
    'both_medium': all_both_medium,
    'both_high': all_both_high,
    'low_medium': all_low_medium,
    'low_high': all_low_high,
    'medium_high': all_medium_high,
    'pct_both_low': (all_both_low/total_all*100) if total_all > 0 else 0,
    'pct_both_medium': (all_both_medium/total_all*100) if total_all > 0 else 0,
    'pct_both_high': (all_both_high/total_all*100) if total_all > 0 else 0,
    'pct_low_medium': (all_low_medium/total_all*100) if total_all > 0 else 0,
    'pct_low_high': (all_low_high/total_all*100) if total_all > 0 else 0,
    'pct_medium_high': (all_medium_high/total_all*100) if total_all > 0 else 0,
}

# Print results in a nicely formatted table
print("Detailed Analysis of Closing Bet Patterns by Experiment Type")
print("=" * 100)
headers = [
    "Exp Type", "Total",
    "Both ≤50%", "Both 51-75%", "Both >75%",
    "50%+51-75%", "50%+>75%", "51-75%+>75%"
]

print(f"{headers[0]:<15} {headers[1]:<7} {headers[2]:<10} {headers[3]:<12} {headers[4]:<10} {headers[5]:<12} {headers[6]:<10} {headers[7]:<12}")
print("-" * 100)

# First print individual experiment types (except overall)
for exp_type in ['cross_model', 'self_debate', 'informed_self', 'public_bets']:
    data = results[exp_type]
    print(f"{exp_type:<15} {data['total']:<7} "
          f"{data['pct_both_low']:.1f}% {data['pct_both_medium']:.1f}% {data['pct_both_high']:.1f}% "
          f"{data['pct_low_medium']:.1f}% {data['pct_low_high']:.1f}% {data['pct_medium_high']:.1f}%")

# Then print overall as the last line
print("-" * 100)
data = results['overall']
print(f"{'overall':<15} {data['total']:<7} "
      f"{data['pct_both_low']:.1f}% {data['pct_both_medium']:.1f}% {data['pct_both_high']:.1f}% "
      f"{data['pct_low_medium']:.1f}% {data['pct_low_high']:.1f}% {data['pct_medium_high']:.1f}%")

# Print raw counts
print("\nRaw Counts:")
print("=" * 100)
print(f"{headers[0]:<15} {headers[1]:<7} {headers[2]:<10} {headers[3]:<12} {headers[4]:<10} {headers[5]:<12} {headers[6]:<10} {headers[7]:<12}")
print("-" * 100)

# First print individual experiment types (except overall)
for exp_type in ['cross_model', 'self_debate', 'informed_self', 'public_bets']:
    data = results[exp_type]
    print(f"{exp_type:<15} {data['total']:<7} "
          f"{data['both_low']:<10} {data['both_medium']:<12} {data['both_high']:<10} "
          f"{data['low_medium']:<12} {data['low_high']:<10} {data['medium_high']:<12}")

# Then print overall as the last line
print("-" * 100)
data = results['overall']
print(f"{'overall':<15} {data['total']:<7} "
      f"{data['both_low']:<10} {data['both_medium']:<12} {data['both_high']:<10} "
      f"{data['low_medium']:<12} {data['low_high']:<10} {data['medium_high']:<12}")
