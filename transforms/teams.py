import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('../data-csv/teams.csv')

# Function to generate instruction-response pairs
def generate_instruction_response_pairs(row):
    pairs = []

    # General team information
    pairs.append({
        'prompt': f"<s>[INST] What is the name of the team with the code {row['team_code']}? [/INST]",
        'completion': f"The name of the team with the code {row['team_code']} is {row['team_name']}."
    })
    
    pairs.append({
        'prompt': f"<s>[INST] Which federation does the team {row['team_name']} belong to? [/INST]",
        'completion': f"The team {row['team_name']} belongs to the federation {row['federation_name']}."
    })
    
    pairs.append({
        'prompt': f"<s>[INST] In which region is the team {row['team_name']} located? [/INST]",
        'completion': f"The team {row['team_name']} is located in the region {row['region_name']}."
    })
    
    pairs.append({
        'prompt': f"<s>[INST] What is the confederation of the team {row['team_name']}? [/INST]",
        'completion': f"The confederation of the team {row['team_name']} is {row['confederation_name']} abbreviated in {row['confederation_code']}."
    })

 #   pairs.append({
 #       'prompt': f"<s>[INST] What is the abbreviation for the confederation of the team {row['team_name']}? [/INST]",
 #       'completion': f"The abbreviation for the confederation of the team {row['team_name']} is {row['confederation_code']}."
 #   })

    # Men's and Women's team qualification information
    if row['mens_team']:
        pairs.append({
            'prompt': f"<s>[INST] Has the men's team of {row['team_name']} qualified for a World Cup tournament? [/INST]",
            'completion': f"Yes, the men's team of {row['team_name']} has qualified for a World Cup tournament."
        })
    else:
        pairs.append({
            'prompt': f"<s>[INST] Has the men's team of {row['team_name']} qualified for a World Cup tournament? [/INST]",
            'completion': f"No, the men's team of {row['team_name']} has not qualified for a World Cup tournament."
        })

    if row['womens_team']:
        pairs.append({
            'prompt': f"<s>[INST] Has the women's team of {row['team_name']} qualified for a World Cup tournament? [/INST]",
            'completion': f"Yes, the women's team of {row['team_name']} has qualified for a World Cup tournament."
        })
    else:
        pairs.append({
            'prompt': f"<s>[INST] Has the women's team of {row['team_name']} qualified for a World Cup tournament? [/INST]",
            'completion': f"No, the women's team of {row['team_name']} has not qualified for a World Cup tournament."
        })

    return pairs

# Generate pairs for each entry in the DataFrame
instruction_response_pairs = []
for _, row in df.iterrows():
    instruction_response_pairs.extend(generate_instruction_response_pairs(row))

# Convert to DataFrame for easy handling
pairs_df = pd.DataFrame(instruction_response_pairs)

# Add the 'train' column with 'evaluation' assigned to 15% of the entries and 'train' to the remaining 80%
pairs_df['train'] = np.where(np.random.rand(len(pairs_df)) < 0.15, 'evaluation', 'train')

# Rename the columns to 'prompt' and 'completion'
pairs_df = pairs_df.rename(columns={'prompt': 'prompt', 'completion': 'completion'})

# Create a replica for the entries labeled "evaluation" with the value "train" for the "train" column
eval_df = pairs_df[pairs_df['train'] == 'evaluation'].copy()
eval_df['train'] = 'train'

# Combine the original pairs_df with the replicated evaluation entries
final_df = pd.concat([pairs_df, eval_df])

# Save to CSV
final_df.to_csv('teams_pairs.csv', index=False)


print("Instruction-response pairs generated and saved to 'teams_pairs.csv'.")