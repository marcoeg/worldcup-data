import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data-csv/goals.csv')

# Function to generate prompt and completion pairs
def generate_instruction_response(row):
    goal_type = "regular play"
    if row['own_goal'] == 1:
        goal_type = "an own goal"
    elif row['penalty'] == 1:
        goal_type = "a penalty"

    prompt = (
        f"Provide details for the goal scored by {row['given_name']} {row['family_name']} of {row['team_name']} in the {row['tournament_name']} match {row['match_name']} "
        #f"{'home team' if row['home_team'] == 1 else 'away team'} {row['team_name']} and their opponent "
        f"on {row['match_date']} at the {row['stage_name']} stage. "
        #f"The goal was scored by {row['given_name']} {row['family_name']} of {row['team_name']} "
        #f"wearing shirt number {row['shirt_number']} in the {row['minute_label']} minute ({row['match_period']}). "
        #f"The goal was {goal_type}."
    )
    
    shirt = ""
    if (row['shirt_number'] != 0):
     shirt = f"wearing shirt number {row['shirt_number']} "

    response = (
        f"The goal was scored by {row['given_name']} {row['family_name']} of {row['team_name']} in the {row['minute_label']} minute ({row['match_period']}) {shirt}"
        f"during the {row['tournament_name']} match {row['match_name']} on {row['match_date']} at the {row['stage_name']} stage. "
        f"It was {goal_type}. "
        f"Minute of regulation: {row['minute_regulation']}, minute of stoppage: {row['minute_stoppage']}."
    )
    
    return f"<s>[INST] {prompt} [/INST]", response

# Generate the instruction-response pairs
pairs = df.apply(generate_instruction_response, axis=1)

# Create a DataFrame for the pairs
pairs_df = pd.DataFrame(pairs.tolist(), columns=['prompt', 'completion'])

# Add the 'train' column with 80% entries as 'train' and 20% as 'evaluation'
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.8, 'train', 'evaluation')

# Create a replica for the entries labeled "evaluation" with the value "train" for the "train" column
eval_df = pairs_df[pairs_df['train'] == 'evaluation'].copy()
eval_df['train'] = 'train'

# Combine the original pairs_df with the replicated evaluation entries
final_df = pd.concat([pairs_df, eval_df])

# Save to CSV
final_df.to_csv('goals_pairs.csv', index=False)

print("Instruction-response pairs have been generated and saved to 'goals_pairs.csv'.")