import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('../data-csv/players.csv')

# Function to generate instruction-response pairs
def generate_instruction_response_pairs(row):
    pairs = []

    # General player information
    #pairs.append({
    #    'prompt': f"<s>[INST] What is the full name of the player with ID {row['player_id']}? [/INST]",
    #    'completion': f"The full name of the player with ID {row['player_id']} is {row['given_name']} {row['family_name']}."
    #})

    pairs.append({
        'prompt': f"<s>[INST] When was the world cup player {row['given_name']} {row['family_name']} born? [/INST]",
        'completion': f"{row['given_name']} {row['family_name']} was born on {row['birth_date']}."
    })

    pairs.append({
        'prompt': f"<s>[INST] Is {row['given_name']} {row['family_name']} male or female? [/INST]",
        'completion': f"{row['given_name']} {row['family_name']} is {'female' if row['female'] else 'male'}."
    })

    pairs.append({
        'prompt': f"<s>[INST] How many World Cup tournaments did {row['given_name']} {row['family_name']} participate in? [/INST]",
        'completion': f"{row['given_name']} {row['family_name']} participated in {row['count_tournaments']} World Cup tournaments."
    })

    from datetime import datetime

    # Given date of birth and years as a string
    dob = row['birth_date']
    if (dob != 'not available') and (not pd.isna(dob)):
        years_str = row['list_tournaments']
        
        # Convert the date of birth to a datetime object
        # Check if the date of birth is in the correct format
        try:
            dob_date = datetime.strptime(dob, "%Y-%m-%d")
        except:
            dob_date = None
            print(row)
            print(f"error in dob {dob}")

        # Split the years string into a list of years
        years = years_str.split(", ")

        # Generate the new list with ages
        ages_list = [f"{year} at age {int(year) - dob_date.year}" for year in years]

        pairs.append({
            'prompt': f"<s>[INST] Which World Cup tournaments did {row['given_name']} {row['family_name']} participate in? [/INST]",
            'completion': f"{row['given_name']} {row['family_name']} participated in the following tournaments: {ages_list}."
        })
    else:
        pairs.append({
            'prompt': f"<s>[INST] Which World Cup tournaments did {row['given_name']} {row['family_name']} participate in? [/INST]",
            'completion': f"{row['given_name']} {row['family_name']} participated in the following tournaments: {row['list_tournaments']}."
        })

    # Player position information
    if row['goal_keeper']:
        pairs.append({
            'prompt': f"<s>[INST] Was {row['given_name']} {row['family_name']} a goal keeper? [/INST]",
            'completion': f"Yes, {row['given_name']} {row['family_name']} was a goal keeper."
        })

    if row['defender']:
        pairs.append({
            'prompt': f"<s>[INST] Was {row['given_name']} {row['family_name']} a defender? [/INST]",
            'completion': f"Yes, {row['given_name']} {row['family_name']} was a defender."
        })

    if row['midfielder']:
        pairs.append({
            'prompt': f"<s>[INST] Was {row['given_name']} {row['family_name']} a midfielder? [/INST]",
            'completion': f"Yes, {row['given_name']} {row['family_name']} was a midfielder."
        })

    if row['forward']:
        pairs.append({
            'prompt': f"<s>[INST] Was {row['given_name']} {row['family_name']} a forward? [/INST]",
            'completion': f"Yes, {row['given_name']} {row['family_name']} was a forward."
        })

    # Wikipedia link information
    #if row['player_wikipedia_link'] != "not applicable":
    #    pairs.append({
    #        'prompt': f"<s>[INST] Where can I find more information about {row['given_name']} {row['family_name']}? [/INST]",
    #        'completion': f"You can find more information about {row['given_name']} {row['family_name']} at {row['player_wikipedia_link']}."
    #    })

    return pairs

# Generate pairs for each entry in the DataFrame
instruction_response_pairs = []
for _, row in df.iterrows():
    instruction_response_pairs.extend(generate_instruction_response_pairs(row))

# Convert to DataFrame for easy handling
pairs_df = pd.DataFrame(instruction_response_pairs)

# Add the 'train' column with 'evaluation' assigned to 20% of the entries and 'train' to the remaining 80%
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.20, 'evaluation', 'train')

# Rename the columns to 'prompt' and 'completion'
pairs_df = pairs_df.rename(columns={'prompt': 'prompt', 'completion': 'completion'})


# Create a replica for the entries labeled "evaluation" with the value "train" for the "train" column
eval_df = pairs_df[pairs_df['train'] == 'evaluation'].copy()
eval_df['train'] = 'train'

# Combine the original pairs_df with the replicated evaluation entries
final_df = pd.concat([pairs_df, eval_df])

# Save to CSV
final_df.to_csv('players_pairs.csv', index=False)

print("Instruction-response pairs generated and saved to 'players_pairs.csv'.")