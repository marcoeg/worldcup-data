import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('../data-csv/tournaments.csv')

# Function to generate instruction-response pairs
def generate_instruction_response_pairs(row):
    pairs = []

    # General tournament information
    pairs.append({
        'prompt': f"In which year was the {row['tournament_name']} held?",
        'completion': f"The {row['tournament_name']} tournament was held in the year {row['year']}."
    })

    pairs.append({
        'prompt': f"Which country hosted the {row['tournament_name']} tournament in {row['year']}?",
        'completion': f"The host country of the {row['tournament_name']} tournament in {row['year']} was {row['host_country']}."
    })

    pairs.append({
        'prompt': f"When did the {row['tournament_name']} tournament start?",
        'completion': f"The {row['tournament_name']} tournament started on {row['start_date']}."
    })
    pairs.append({
        'prompt': f"When did the {row['tournament_name']} tournament end?",
        'completion': f"The {row['tournament_name']} tournament ended on {row['end_date']}."
    })
    pairs.append({
        'prompt': f"Which team won the {row['tournament_name']} tournament in {row['year']}?",
        'completion': f"The team that won the {row['tournament_name']} tournament in {row['year']} was {row['winner']}."
    })

    # Tournament format information
    if row['host_won']:
        pairs.append({
            'prompt': f"Did the host country win the {row['tournament_name']} tournament in {row['year']}?",
            'completion': f"Yes, the host country won the {row['tournament_name']} tournament in {row['year']}."
        })
    else:
        pairs.append({
            'prompt': f"Did the host country win the {row['tournament_name']} tournament in {row['year']}?",
            'completion': f"No, the host country did not win the {row['tournament_name']} tournament in {row['year']}."
        })

    pairs.append({
        'prompt': f"How many teams participated in the {row['tournament_name']} tournament in {row['year']}?",
        'completion': f"The number of teams that participated in the {row['tournament_name']} tournament was {row['count_teams']}."
    })

    format_stages = [
        ('group_stage', 'group stage'),
        ('second_group_stage', 'second group stage'),
        ('final_round', 'final round'),
        ('round_of_16', 'round of 16 stage'),
        ('quarter_finals', 'quarter-finals stage'),
        ('semi_finals', 'semi-finals stage'),
        ('third_place_match', 'third-place match'),
        ('final', 'final match')
    ]

    for stage, stage_name in format_stages:
        if row[stage]:
            pairs.append({
                'prompt': f"Was there a {stage_name} in the {row['tournament_name']} tournament of {row['year']}?",
                'completion': f"Yes, there was a {stage_name} in the {row['tournament_name']} tournament of {row['year']}."
            })
        else:
            pairs.append({
                'prompt': f"Was there a {stage_name} in the {row['tournament_name']} tournament of {row['year']}?",
                'completion': f"No, there was not a {stage_name} in the {row['tournament_name']} tournament of {row['year']}."
            })

    return pairs

# Generate pairs for each entry in the DataFrame
instruction_response_pairs = []
for _, row in df.iterrows():
    instruction_response_pairs.extend(generate_instruction_response_pairs(row))

# Convert to DataFrame for easy handling
pairs_df = pd.DataFrame(instruction_response_pairs)

# Add the 'train' column with 'evaluation' assigned to none of the entries
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.0, 'evaluation', 'train')

# Rename the columns to 'prompt' and 'completion'
pairs_df = pairs_df.rename(columns={'prompt': 'prompt', 'completion': 'completion'})

# Save to a CSV file
pairs_df.to_csv('tournaments_pairs.csv', index=False)

print("Instruction-response pairs generated and saved to 'tournaments_pairs.csv'.")