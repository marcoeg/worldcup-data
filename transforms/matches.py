import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data-csv/matches.csv')

# Function to generate prompt and completion pairs
def generate_instruction_response(row):
    # Base prompt
    prompt = (
        f"Provide details for the {row['tournament_name']} match between {row['home_team_name']} and {row['away_team_name']} "
        f"held on {row['match_date']} at {row['stadium_name']} in {row['city_name']}, {row['country_name']}. "
    )
    
    # Determine match stage
    match_stage = ""
    if row['group_stage'] == 1:
        match_stage = f"The stage of the match was {row['stage_name']}, {row['group_name']}. "
    elif row['knockout_stage'] == 1:
        match_stage = f"The match was a knockout stage. "

    # Determine the winning team
    winning_team = ""
    if row['home_team_win'] == 1:
        winning_team = f"Winning team was {row['home_team_name']}."
    elif row['away_team_win'] == 1:
        winning_team = f"Winning team was {row['away_team_name']}."
    
    # Extra time and penalty shootout details
    extra_time_details = f" {'Extra time: Yes.' if row['extra_time'] == 1 else ''}"
    penalty_shootout_details = ""
    if row['penalty_shootout'] == 1:
        penalty_shootout_details = (
            f"The match ended in a penalty shootout with a score of {row['score_penalties']}. "
            f"Penalties scored: {row['home_team_name']} {row['home_team_score_penalties']}, "
            f"{row['away_team_name']} {row['away_team_score_penalties']}."
        )
    
    # Complete prompt
    prompt += match_stage # + extra_time_details + penalty_shootout_details

    # Response
    response = (
        f"The match was between {row['home_team_name']} and {row['away_team_name']} for the {row['tournament_name']}. "
        f"It was held on {row['match_date']} at {row['stadium_name']} in {row['city_name']}, {row['country_name']}. "
        f"{match_stage}"
        f"The final score was {row['score']}. "
        f"The match result was a {row['result']}. {winning_team} "
        f"{extra_time_details}{penalty_shootout_details}"
    )
    return f"<s>[INST] {prompt} [/INST]", response

# Generate the instruction-response pairs
pairs = df.apply(generate_instruction_response, axis=1)

# Create a DataFrame for the pairs
pairs_df = pd.DataFrame(pairs.tolist(), columns=['prompt', 'completion'])

"""
# Add the 'train' column with 80% entries as 'train' and 20% as 'evaluation'
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.80, 'train', 'evaluation')

# Create a replica for the entries labeled "evaluation" with the value "train" for the "train" column
eval_df = pairs_df[pairs_df['train'] == 'evaluation'].copy()
eval_df['train'] = 'train'

# Combine the original pairs_df with the replicated evaluation entries
final_df = pd.concat([pairs_df, eval_df])

# Save to CSV
final_df.to_csv('matches_pairs.csv', index=False)
"""

# Save to CSV
pairs_df.to_csv('matches_pairs.csv', index=False)

print("Instruction-response pairs have been generated and saved to 'matches_pairs.csv'.")