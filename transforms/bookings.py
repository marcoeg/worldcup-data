import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data-csv/bookings.csv')

# Function to generate prompt and completion pairs
def generate_instruction_response(row):
    # Determine the appropriate minute detail
    if row['minute_stoppage'] == 0:
        minute_detail = f"minute {row['minute_regulation']}"
    else:
        minute_detail = f"minute {row['minute_stoppage']} of stoppage time"

    # Construct the prompt
    prompt = (
        f"Provide details for the booking in the {row['tournament_name']} match {row['match_name']} "
        f"on {row['match_date']} at the {row['stage_name']} stage. "
        f"The booking was for {row['given_name']} {row['family_name']} of {row['team_name']}."
    )

    # Construct the completion
    shirt_number_detail = f"wearing shirt number {row['shirt_number']} " if row['shirt_number'] != 0 else ""
    yellow_card_detail = "received a yellow card. " if row['yellow_card'] == 1 else ""
    second_yellow_card_detail = f" It was the second yellow card." if row['second_yellow_card'] == 1 else ""
    red_card_detail = "received a red card. " if row['red_card'] == 1 else ""
    sending_off_detail = f"The player was sent off." if row['sending_off'] == 1 else ""

    completion = (
        f"The booking was for {row['given_name']} {row['family_name']} of {row['team_name']} "
        f"{shirt_number_detail}during the {row['tournament_name']} match {row['match_name']} on {row['match_date']} "
        f"at the {row['stage_name']} stage in the {row['match_period']}. The player at the {minute_detail} "
        f"{yellow_card_detail}{second_yellow_card_detail} {red_card_detail}{sending_off_detail}"
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
final_df.to_csv('bookings_pairs.csv', index=False)
"""
# Save to CSV
pairs_df.to_csv('bookings_pairs.csv', index=False)

print("Instruction-response pairs have been generated and saved to 'bookings_pairs.csv'.")