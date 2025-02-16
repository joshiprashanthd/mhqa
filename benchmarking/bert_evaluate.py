import sys
sys.path.append("..")

import re
import argparse
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer,AutoModelForMultipleChoice
import functools
import torch
from tqdm import tqdm
import os
from pathlib import Path
from sft.dataset import MCQADataset
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
def process_batch(batch, tokenizer, max_len=32):
    expanded_batch = []
    labels = []
    batch_size = len(batch)
    num_choices = 4
    for data_tuple in batch:
        question,options,label = data_tuple
        expanded_batch += [[str(question), str(option)] for option in options]
        labels.append(label)
    
    try:
        tokenized_batch = tokenizer.batch_encode_plus(expanded_batch, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        input_ids = tokenized_batch["input_ids"].view(batch_size, num_choices, -1)
        attention_mask = tokenized_batch["attention_mask"].view(batch_size, num_choices, -1)
        tokenized_batch.update({"input_ids": input_ids, "attention_mask": attention_mask})
        return tokenized_batch, torch.tensor(labels)
    except Exception as e:
        print(e)
        pprint(expanded_batch)
        sys.exit(1)
  
def create_dataloader(dataset, tokenizer, batch_size, max_len):
    eval_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        process_batch,
        tokenizer=tokenizer,
        max_len=max_len
        )
    test_dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn, 
                                num_workers=1)
    return test_dataloader 

def run_inference(model, dataloader):
    predictions = []
    for idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs.logits, axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions

def evaluate(inference_model, tokenizer, experiment_folder, csv_path, batch_size=32, max_len=192):
    dataset = MCQADataset(csv_path)
    dataloader = create_dataloader(dataset, tokenizer, batch_size, max_len)
    test_df = pd.read_csv(csv_path)
    test_df.loc[:, "predictions"] = [pred+1 for pred in run_inference(inference_model, dataloader)]
    test_df.to_csv(os.path.join(experiment_folder,"test_results.csv"),index=False)
    print(f"Test predictions written to {os.path.join(experiment_folder,'test_results.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="name of the model")
    parser.add_argument("--csv_path", help="path to the csv file", required=True) 
    cmd_args = parser.parse_args()

    experiment_name = re.sub(r"[\.|\/|\-]", "_", cmd_args.csv_path)
    print(f"Evaluating model {cmd_args.model} on {experiment_name}")

    model = cmd_args.model
    
    exp_name = Path(f"{model}{experiment_name}".replace("/","_"))

    inference_model = AutoModelForMultipleChoice.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    inference_model = inference_model.to(device)

    experiment_folder = Path("logs") / exp_name
    experiment_folder.mkdir(parents=True, exist_ok=True)

    evaluate(inference_model, tokenizer, experiment_folder, cmd_args.csv_path)
    