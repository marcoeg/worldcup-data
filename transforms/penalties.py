import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data-csv/penalty_kicks.csv')

# Function to generate prompt and completion pairs
def generate_instruction_response(row):
    # Construct the prompt
    prompt = (
        f"Provide details for the penalty kick taken during the {row['tournament_name']} match {row['match_name']} "
        f"on {row['match_date']} at the {row['stage_name']} stage. "
        f"The penalty was taken by {row['given_name']} {row['family_name']} of {row['team_name']}."
    )
    
    # Construct the completion
    shirt_number_detail = f"wearing shirt number {row['shirt_number']} " if row['shirt_number'] != 0 else ""
    converted_detail = "converted the penalty" if row['converted'] == 1 else "did not convert the penalty"
    
    completion = (
        f"The penalty was taken by {row['given_name']} {row['family_name']} of {row['team_name']} "
        f"{shirt_number_detail}during the {row['tournament_name']} match {row['match_name']} on {row['match_date']} "
        f"at the {row['stage_name']} stage. The player {converted_detail}."
    )
    
    return f"<s>[INST] {prompt} [/INST]", completion

# Generate the instruction-response pairs
pairs = df.apply(generate_instruction_response, axis=1)

# Create a DataFrame for the pairs
pairs_df = pd.DataFrame(pairs.tolist(), columns=['prompt', 'completion'])

"""
# Add the 'train' column with 80% entries as 'train' and 20% as 'evaluation'
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.8, 'train', 'evaluation')

# Create a replica for the entries labeled "evaluation" with the value "train" for the "train" column
eval_df = pairs_df[pairs_df['train'] == 'evaluation'].copy()
eval_df['train'] = 'train'

# Combine the original pairs_df with the replicated evaluation entries
final_df = pd.concat([pairs_df, eval_df])

# Save to CSV
final_df.to_csv('penalties_pairs.csv', index=False)
"""

# Save to CSV
pairs_df.to_csv('penalties_pairs.csv', index=False)

print("Instruction-response pairs have been generated and saved to 'penalties_pairs.csv'.")