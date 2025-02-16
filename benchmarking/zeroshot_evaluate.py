import sys
sys.path.append(".")

from logger import CsvLogger
import re, argparse
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
from tqdm import tqdm

from model import BaseModel, OpenAIGPT, HFApiModel

prompt = """
You are a highly skilled medical expert tasked with evaluating multiple-choice questions. 
Your role is to select the single most accurate and contextually correct option from the given choices. 
Do not favor any option based on its order. Evaluate all options carefully and justify your choice implicitly.

Output Format:
Correct Option: <1 or 2 or 3 or 4>
Justification: <justification for the answer>

Question: {question}
Options: 
Option1: {op1}
Option2: {op2}
Option3: {op3}
Option4: {op4}

Respond with the letter corresponding to your choice (1 or 2 or 3 or 4) followed by a justification.
Correct Option: """

csv_lock = Lock()

def extract_answer_justification(s: str):
    match = re.findall(r"[\W]*Correct Option[\W]*: (\d)", s)
    correct_option = int(match[-1].strip()) if len(match) > 0 else None 
    match = re.findall(r"[\W]*Justification[\W]*: (.*)", s)
    justification = match[-1].strip() if len(match) > 0 else None
    if justification == '<justification for the answer>': justification = None
    return (correct_option, justification)

def run_inference(model: BaseModel, df: pd.DataFrame):
    outputs = []
    for idx, row in df.iterrows():
        question = row["question"]
        options = [row[f"option{i}"] for i in range(1, 5)]
        new_prompt = prompt.format(question=question, op1=options[0], op2=options[1], op3=options[2], op4=options[3])
    
        generated_text = model.generate_text(new_prompt)
        print(generated_text)
        answer, justification = extract_answer_justification(generated_text)
        outputs.append([answer, justification, generated_text])
    return outputs

def evaluate(model, batch, reslog):
    outputs = run_inference(model, batch)
    print("Len of outputs:", len(outputs))
    rows = []
    with csv_lock:
        for idx, (_, row) in enumerate(batch.iterrows()):
            rows.append(list(row.values) + outputs[idx])
        reslog.logrows(rows)

if __name__ == "__main__":

    openai_model_names = [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4o"
    ]

    parser = argparse.ArgumentParser(description='Evaluate LLM model on multiple choice questions')
    parser.add_argument('--model', type=str, required=True, help='Model name w.r.t the service')
    parser.add_argument("--csv_path", help="path to the csv file", required=True)

    cmd_args = parser.parse_args()

    input_csv_path = Path(cmd_args.csv_path)
    log_folder_path = Path("./logs")
    log_folder_path.mkdir(parents=True, exist_ok=True)
    sluggify = re.sub(r"[\.|\/|\-]", "_", cmd_args.csv_path)
    exp_name = Path(f"{cmd_args.model}_{sluggify}".replace("/","_"))
    experiment_folder_path = log_folder_path / exp_name
    experiment_folder_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cmd_args.model in openai_model_names:
        print("Using OpenAI GPT model")
        model = OpenAIGPT(cmd_args.model, device)
    else: 
        print("Using HuggingFace model")
        model = HFApiModel(cmd_args.model, device)

    df = pd.read_csv(input_csv_path) 
    header = list(df.columns)

    result_logger = CsvLogger(experiment_folder_path, "test_results", header + ["predictions", "justification", "response"])

    batch_size = 2

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            futures.append(executor.submit(evaluate, model, batch, result_logger))
        for future in tqdm(futures):
            future.result()



