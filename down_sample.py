import os
import pandas as pd
import numpy as np
import sys
import argparse

# Get input and output folder
parser = argparse.ArgumentParser(description='Down sample SST-2 data')

parser.add_argument('--input-path', '-i', required=True, type=str,
    help='input directory path for SST-2 data.')
parser.add_argument('--output-path', '-o', required=True, type=str,
    help='output directory path for down-sampled data.')

args = parser.parse_args()

input_path = args.inpu_path
output_path = args.output_path

# Set the random seed
random_seed = 42

# Load data
train = pd.read_csv(os.path.join(input_path,'train.tsv'),sep='\t')
test = pd.read_csv(os.path.join(input_path,'dev.tsv'),sep='\t')

# Rename columns
train.rename(columns={'sentence':'text'}, inplace=True)
test.rename(columns={'sentence':'text'}, inplace=True)

# Imbalance ratios
rates = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Down sample function
def down_samp(df, rate, random_seed):
    
    df_pos = df.loc[df['label']==1,:]
    df_neg = df.loc[df['label']==0,:]
    
    # Number of negative samples to keep:
    num_neg_to_keep = int(np.floor(len(df_pos)*(rate)))
    
    # Decimated negative samples:
    df_neg_down = df_neg.sample(n=num_neg_to_keep, random_state=random_seed)
    
    # Combine and shuflle
    unb_df = pd.concat([df_pos, df_neg_down]).sample(frac=1, random_state=random_seed)
    
    return unb_df
    

for rate in rates:
    
    # Define output path for each rate
    output_path_rate = os.path.join(output_path,str(rate))
    
    # Make directory
    os.makedirs(output_path_rate, exist_ok=True)

    print('\n\nRATE =',rate)

    unb_df = down_samp(train, rate, random_seed)
    unb_df.to_csv(os.path.join(output_path_rate, 'train.csv'), index=False)
    print(unb_df['label'].value_counts())
    
    # Save intact test data
    test.to_csv(os.path.join(output_path_rate, 'test.csv'), index=False)
