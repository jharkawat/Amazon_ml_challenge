import numpy as np 
import pandas as pd
import csv
import os
import glob
import pandas as pd
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification   
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn 
from tqdm import tqdm
import json
from util import f1_score_func, accuracy_per_class
import util
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='roberta-base', help="model name from huggingface")
parser.add_argument('--experiment_name', default='sample-roberta-training-properly', help="model name from huggingface") #sample-try-with-bert-base
parser.add_argument('--used_tokenized_data', type=bool, default=False, help="saved pickled tokenizer faster laoding in future")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size_train", type=int, default=32)
parser.add_argument("--batch_size_val", type=int, default=32)

parser.add_argument("--dummy", type=bool, default=False)
parser.add_argument("--full_finetuning", type=bool, default=True)
parser.add_argument("--loading_from_prev_pretrain", type=bool, default=False)
parser.add_argument("--trained_model", default="sample-distilbert-run2/finetuned_BERT_epoch_1.model")
def evaluate(dataloader_val, model):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(0) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        loss = loss.mean()
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

if (__name__ == "__main__"):

    args = parser.parse_args()

    exp_name = args.experiment_name
    # logging args parsers fileds
    
    #assert Path("./"+exp_name).exists()
    os.mkdir(exp_name)

    util.set_logger(os.path.join(exp_name, 'train.log'))
    logging.info("Training Arguments: {}" .format(args))

    try:
        os.system("nvidia-smi")
    except:
        print("Something went wrong with nvidia-smi command")
    logging.info("loading all the files of data")

    # logging tokeinzer used
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info("Used tokenizer: {}".format(tokenizer))


    # directly loading tokenized data from  from pickle
    if args.used_tokenized_data:
        logging.info("loading tokenized data")
        encoded_data_train = torch.load("data/tokenized/encoded_data_train.pt")
        encoded_data_val = torch.load("data/tokenized/encoded_data_val.pt")

    else:
        #filenames = [name for name in glob.glob('./data/dataset/train.csv')]
        filenames = ["custum-data/val.csv", "custum-data/train.csv"]
        df = pd.concat( [ pd.read_csv(f, low_memory=False)  for f in filenames ] ) 
        dict = {'desc': 'BULLET_POINTS',
        'BROWSE_NODE_ID': 'BROWSE_NODE_ID'}
  
        # call rename () method
        df.rename(columns=dict,
                inplace=True)
  
        
        #df = pd.read_csv("./data/dataset/train.csv", escapechar = "\\", quoting = csv.QUOTE_NONE)
        
        if (args.dummy):
            # logging 
            logging.info("dummy data")
            df = df[:400]

        #df = df[["BULLET_POINTS", "BROWSE_NODE_ID"]]
        #df = pd.concat( [ pd.read_csv(f, sep='\t', names=['id', 'BROWSE_NODE_ID','BULLET_POINTS']) for f in filenames ] )    

        logging.info("Loaded sucessfull")

        possible_labels = df.BROWSE_NODE_ID.unique()
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        
        
        #os.path.join(exp_name, 'params.json')
        with open(os.path.join(exp_name, 'params.json'), 'w') as fp:
            label_dict = {int(k):int(v) for k,v in label_dict.items() }
            json.dump(label_dict, fp)
        # logging the location of dump dict
        logging.info("Dump label_dict location: {}".format(os.path.join(exp_name, 'params.json')))


        df['label'] = df.BROWSE_NODE_ID.replace(label_dict)
        df = df.dropna()
        # drop row if no of label in BROWSE_NODE_ID is less than 2

        X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size=0.35, 
                                                    random_state=44)

        df['data_type'] = ['not_set']*df.shape[0]
        df.loc[X_train, 'data_type'] = 'train'
        df.loc[X_val, 'data_type'] = 'val'


        encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].BULLET_POINTS.values.tolist(), 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
        )
        # logging of encoded data
        logging.info("Encoded train data: encoded_data_train")

        encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].BULLET_POINTS.values.tolist(), 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
        )
        logging.info("Encoded val data: encoded_data_val")
        #torch.save(encoded_data_train, exp_name+"/encoded_data_train.pt") 
        #torch.save(encoded_data_val, exp_name+"/encoded_data_val.pt")
        
        logging.info("Dumped encoded_data_train.pt and encoded_data_val.pt")

    
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


 ## BERT MODEL
    logging.info("Loading AutoModel model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
    if args.loading_from_prev_pretrain: 
        logging.info("Loading pretrained model")
        model.load_state_dict(torch.load(args.trained_model))
    
    model = nn.DataParallel(model)
    logging.info("using Multi GPU data parallel")
    
 ## DataLoader
    batch_size_train = args.batch_size_train
    batch_size_val = args.batch_size_val
    logging.info("batch size train: {}" .format(batch_size_train))
    logging.info("batch size val: {}" .format(batch_size_val))
    dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size_train)
    dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size_val)

    #optimizer = AdamW(model.parameters(),
    #             lr=1e-5, 
    #             eps=1e-8)
    
    epochs = args.epochs
    weight_decay = 0.01
    logging.info("epochs: {}" .format(epochs))
    
    if args.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
    
    # embedding(tokenzing instialled form based pretrained)--> Architecture (tranformer based stacks) --> classifier (linear classifier/ svm ) # intally ended to trainabl e
    else: # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    model = model.to(0)


    logging.info("Device: {}" .format(device))
    #print(device)
    best_acc = 0.0
    patience_counter = 0

    # logging all the paramters of agrs
    #logging.info("Training Arguments: {}" .format(args))
    for epoch in tqdm(range(1, epochs+1)):
    
        if args.full_finetuning:
            model.train()
        else:
            model.classifier.train()
    
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
        
            batch = tuple(b.to(0) for b in batch)
        
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

            outputs = model(**inputs)
        
            loss = outputs[0]
            loss = loss.mean()
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
       
        torch.save(model.state_dict(), f'{exp_name}/finetuned_BERT_epoch_{epoch}.model')
        
        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
        val_f1, acc = f1_score_func(predictions, true_vals)
        
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        logging.info(f'F1 Score (Weighted): {val_f1}')
        logging.info(f'Accuracy: {acc}')
        
        improve_acc = acc - best_acc
        patience = 0.02
        patience_num = 10
        min_epoch_num =5
        
        if improve_acc > 1e-5:
            logging.info("- Found new best Accuarcy")
            best_acc = acc
            torch.save(model.state_dict(), f'{exp_name}/finetuned_BERT_best.model')
            if improve_acc < patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
            
                    # Early stopping and logging best f1
        if (patience_counter >= patience_num and epoch > min_epoch_num) or epoch == epochs:
            logging.info("Best val f1: {:05.2f}".format(best_acc))
            break

