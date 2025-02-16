import functools
import sys
sys.path.append(".")

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from model import MCQAModel
from dataset import MCQADataset
from sklearn.metrics import classification_report


def process_batch(batch, tokenizer, max_len=32):
    expanded_batch = []
    labels = []
    for data_tuple in batch:
        question,options,label = data_tuple
        question_option_pairs = [question+' '+option for option in options]
        labels.append(label)
        expanded_batch.extend(question_option_pairs)
    tokenized_batch = tokenizer.batch_encode_plus(expanded_batch, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    return tokenized_batch, torch.LongTensor(labels)

def get_dataloader(dataset, tokenizer, args):
    eval_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
      process_batch,
      tokenizer=tokenizer,
      max_len=args['max_len']
    )
    test_dataloader = DataLoader(dataset,
                                batch_size=args['batch_size'],
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn, 
                                num_workers=95)
    return test_dataloader

def run_inference(model, dataloader, args):
    predictions = []
    for idx, (inputs,labels) in tqdm(enumerate(dataloader)):
        batch_size = len(labels)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args['device'])
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    cmd_args = parser.parse_args()

    ckpt_path = Path(cmd_args.ckpt_path)
    csv_path = Path(cmd_args.csv_path)

    model = MCQAModel.load_from_checkpoint(ckpt_path)
    model = model.to("cuda")

    dataset = MCQADataset(cmd_args.csv_path)
    dataloader = get_dataloader(dataset, model.tokenizer, model.args)

    args = {
        "batch_size": 32,
        "max_len": 192,
        "device": "cuda"
    }

    logs = Path("./logs")
    results_path = logs / ckpt_path.parent.stem
    results_path.mkdir(parents=True, exist_ok=True)
    df_save_path = results_path / "test_results.csv"
    report_save_path = results_path / "classification_report.txt"

    df = pd.read_csv(csv_path)
    df.loc[:, "predictions"] = [pred+1 for pred in run_inference(model, dataloader, args)]
    df.to_csv(df_save_path, index=False)

    y_true = df["correct_option_number"].astype(int)
    y_pred = df["predictions"].astype(int)
    report = classification_report(y_true, y_pred)
    with open(report_save_path, "w") as f:
        f.write(str(report))


    
