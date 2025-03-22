import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from time import time
from sklearn.metrics import f1_score
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler)
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    logging)


"""Get arguments"""

parser = argparse.ArgumentParser(description='Down sample SST-2 data')

parser.add_argument('--data-path', '-d', required=True, type=str,
    help='input directory path for SST-2 data.')
parser.add_argument('--output-path', '-o', required=True, type=str,
    help='output directory path for down-sampled data.')
parser.add_argument('--epoch-stage-1', '-e', required=True, type=int,
    help='Number of epochs in Stage 1', default=1)
parser.add_argument('--epoch-stage-2', '-f', required=True, type=int,
    help='Number of epochs in Stage 2', default=1)

args = parser.parse_args()

data_path = args.data_path
output_folder = args.output_path
epoch_stage_1 = args.epoch_stage_1
epoch_stage_2 = args.epoch_stage_2


"""Hyperparameters"""

max_len = 128
batch_size = 16
lr_stage_1 = 1e-4
lr_stage_2 = 1e-5
num_stage_1_train_layers = 1 # Number of trained layers in stage 1
save_per_step = 200
validate_during_training = True
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
num_labels = 2
seed_list = [7, 42, 106, 190, 191]


"""Load data and model"""

train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
aug_df = pd.read_csv(os.path.join(data_path,'aug.csv'))
test_df = pd.read_csv(os.path.join(data_path,'test.csv'))

# Creat output folder
os.makedirs(output_folder, exist_ok=True)

# disable warning
logging.set_verbosity_error()

# Load BERT Model
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

filename = 'results.csv'


"""## Create DataLoader"""

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.text = df['text'].values
        self.label = df['label'].values
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        # text = self.text[index]
        # text = ' '.join(text.split())
        text = str(self.text[index])
        label = self.label[index]

        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False, # True 
            max_length=max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }


"""Build BERT Model"""

class BertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H = 768, 50
        
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_attentions=True, output_hidden_states=True)
        # whether the model returns ALL hidden states
        
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(nn.Linear(D_in, H),
                                    # nn.Tanh(),
                                    nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    nn.Linear(H, num_labels))

    def forward(self, input_ids, attention_mask):
        # Encode input text. outputs: a tuple of (last_hidden_state, pooler_output)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        pooler_output = outputs[1] 
        logits = self.classifier(pooler_output)

        return logits


"""Prepare data"""

dataset_train = BERTDataset(train_df)
dataset_aug = BERTDataset(aug_df)
dataset_val = BERTDataset(test_df)

dataloader_train = DataLoader(dataset_train, 
                                 sampler=RandomSampler(dataset_train), 
                                 batch_size=batch_size)

dataloader_aug = DataLoader(dataset_aug, 
                                 sampler=RandomSampler(dataset_aug), 
                                 batch_size=batch_size)

dataloader_val = DataLoader(dataset_val,
                               sampler=SequentialSampler(dataset_val), 
                               batch_size=batch_size)


"""Evaluator"""

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for step, batch in enumerate(dataloader_val):

        inputs = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():        
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
        # Compute loss
        # loss = loss_fn(logits, b_labels)
        loss = loss_fn_stage_2(logits, inputs['labels'])
        loss_val_total += loss.item()
        
        # Get the predictions
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


"""Run the experiments"""

result_df = pd.DataFrame(columns=['Seed', 'F1', 'F1_0', 'F1_1'])

# Run the experimetn for different seeds
for seed in seed_list:
    
    print('Seed:',seed)
    
    # Start time and load seed
    start = time()
    total_step = 0
    set_seed(seed)
    
    # Load the model
    model = BertClassifier(num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Stage 1
    
    # Freeze the model
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    for name, param in model.classifier.named_parameters(): 
        param.requires_grad = True
    
    # Unfreeze pooler
    for name, param in model.bert.pooler.named_parameters(): 
        param.requires_grad = True

    # Unfreeze only topmost layers of BERT
    for name, param in model.bert.encoder.layer[0-num_stage_1_train_layers:].named_parameters(): 
        param.requires_grad = True

    # Optimizer
    optimizer_stage_1 = AdamW(model.parameters(),
                            lr=lr_stage_1, 
                            eps=1e-8)
    
    # Scheduler
    scheduler_stage_1 = get_linear_schedule_with_warmup(optimizer=optimizer_stage_1,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train)*epoch_stage_1)
    
    
    #Specify loss function
    loss_stage_1 = nn.CrossEntropyLoss()

    # =============Stage 1==================
    for epoch in range(epoch_stage_1):
        
        print(f'\nStage 1. Epoch {epoch}')
        
        model.train()
        
        loss_train_total = 0

        for step, batch in enumerate(dataloader_aug):
            
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            ## Compute loss and accumulate the loss values
            loss = loss_stage_1(logits, inputs['labels'])
            loss_train_total += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage_1.step()
            scheduler_stage_1.step()

            total_step += 1


    # =============Stage 2==================

    # Unfreeze 
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer_stage_2 = AdamW(model.parameters(),
                         lr=lr_stage_2, 
                         eps=1e-8)
    
    # Scheduler
    scheduler_stage_2 = get_linear_schedule_with_warmup(optimizer=optimizer_stage_2, 
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train)*epoch_stage_2)

    #######################
    # Specify loss function
    loss_fn_stage_2 = nn.CrossEntropyLoss()

    for epoch in range(epoch_stage_2):
        
        print(f'\nStage 2. Epoch {epoch}')
        
        model.train()
        loss_train_total = 0

        for step, batch in enumerate(dataloader_train):
            
            model.train()
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            

            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            ## Compute loss and accumulate the loss values
            loss = loss_fn_stage_2(logits, inputs['labels'])
            loss_train_total += loss.item()
        
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer_stage_2.step()
            scheduler_stage_2.step()

            total_step += 1


    print('Testing')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    f1 = f1_score(true_vals, preds_flat, average='micro')
    f1_class = f1_score(true_vals, preds_flat, average=None)
    runtime = time()-start
    result_df = result_df.append({
        'Seed': seed,
        'F1': f1,
        'F1_0': f1_class[0],
        'F1_1': f1_class[1]
        }, ignore_index=True)
    result_df.to_csv(os.path.join(output_folder, filename), index=True)
    print(f'Running time: {time()-start:.3f} seconds')
    print('F1: ',f1)


"""Average over seeds"""

ave = pd.DataFrame(result_df.mean()).transpose()
ave.to_csv(os.path.join(output_folder, 'ave_sst.csv'), index=True)
