import os
import openai
import argparse
import numpy as np
import pandas as pd
from time import sleep

# Get the arguments
parser = argparse.ArgumentParser(description='Down sample SST-2 data')
parser.add_argument('--data-path', '-d', required=True, type=str,
    help='Directory path for the data.')
parser.add_argument('--api-key', '-k', required=True, type=str,
    help='OpenAI API Key')
args = parser.parse_args()
data_path = args.data_path
api_key = args.api_key

# Set OpenAI key:
openai.api_key = api_key

# Define generate augmented data function:
def gen_aug(text):
    
    # The first part of prompt:
    prompt_start = prompt_start = "Rephrase the following phrase into a different phrase with similar meaning."
    
    # Full prompt:
    prompt = (prompt_start + "\n\n---\n\n" + text)

    response = ""
    
    # If the prompt is too long, the API will not work. We need to make the prompt
    # shorter recursively until it is acceptable by the API
    
    # Cut 10% of the prompt each time.
    len_cut = int(len(prompt)*0.1)
    
    # Set the flag to true so that the loop starts running
    flag = True
    while flag:

        # Try this prompt
        try:
            result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # If no error, do not further run the loop
            flag = False

        # If the prompt is too long, make it shorter
        except:
            prompt = prompt[:-len_cut]
    
    # Extract the reponse
    response = result["choices"][0]["message"]["content"]
    
    return response

# The pipeline can be stopped and continue working later on

# If the file exists, we are continuing augmented data generation:
if os.path.isfile(os.path.join(data_path,'tmp.csv')):

    # Load data
    df = pd.read_csv(os.path.join(data_path,'tmp.csv'))
    data_count = pd.read_csv(os.path.join(data_path,'data_count.csv'))

# If the data does not exist, we start generating data augmentation:
else:

    # Initialize data
    df = pd.read_csv(os.path.join(data_path,'train.csv'))
    df['num_aug_gen'] = 0

    # Data count
    data_count = pd.DataFrame()
    for label in [0, 1]:
        data_count.loc[label, 'original'] = int(df['label'].value_counts()[label])
    data_count['aug'] = 0

# Number of augmented data is the same as the majority class
n_aug = int(np.max(data_count['original']))

# Count errors
error_count = 0

# While there is a class that needs more data
while np.sum(data_count['aug']<n_aug*1.01)>0:
    
    for i in df.index:

        l = df.loc[i,'label']
        
        # If we have generated more than enough data for this class
        if data_count.loc[l, 'aug']>n_aug*1.01:
            continue

        print(i,'/',len(df))

        text = df.loc[i,'text']

        # Delay to bypass rate limitation
        sleep(1)

        try:
            
            # Get the augmented text
            aug_texts = gen_aug(text)

            j =  df.loc[i,'num_aug_gen']
            df.loc[i,'tmp_aug_'+str(j)]=aug_texts

            data_count.loc[l, 'aug']  = data_count.loc[l, 'aug'] + 1
            df.loc[i,'num_aug_gen'] = df.loc[i,'num_aug_gen'] + 1

            print(l, data_count.loc[l, 'aug'])
        
        # If there was an error, wait for 60 seconds
        except:

            # Wait for one minute
            sleep(60)
            
            # Count error
            error_count = error_count + 1
            
            # If we have more than 5 errors, break
            if error_count>5:
                raise Exception('Too many errors')
        
        # Save the data
        df.to_csv(os.path.join(data_path,'tmp.csv'), index=False)
        data_count.to_csv(os.path.join(data_path,'data_count.csv'), index=False)


# Get the list of augmented columns
aug_cols = []
for col in df.columns:
    if col.startswith('tmp_aug'):
       	aug_cols.append(col)

# get index of NaNs
NaNs = df.isna()

all_data = pd.DataFrame(columns=['org_ind', 'paragraph', 'label',])

# Put different augmented samples from the same original sample in different rows
counter = 0
for i in df.index:
    
    l = df.loc[i, 'label']
    
    for col in aug_cols:
       	
        if not NaNs.loc[i, col]:
            
            all_data.loc[counter, 'org_ind'] = i
            all_data.loc[counter, 'paragraph'] = df.loc[i, col]
            all_data.loc[counter, 'label'] = int(l)

            counter += 1

# Final output
out = pd.DataFrame(columns=['org_ind', 'paragraph', 'label',])

for l in set(all_data['label']):
    data_label = all_data.loc[all_data['label']==l, :]
    data_label_sampled = data_label.sample(n=n_aug, random_state=42)
    out = pd.concat([out, data_label_sampled])

# Rename the columns
out.rename(columns={'paragraph': "text"}, inplace=True)

# Shuffle
out_shuff = out.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the data
out_shuff.to_csv(os.path.join(data_path,'aug.csv'), index=False)
