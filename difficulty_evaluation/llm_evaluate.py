import sys
sys.path.append(".")

from logger import CsvLogger
import re, argparse
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from  itertools import zip_longest

import torch
import pandas as pd
from tqdm import tqdm

from model import BaseModel, OpenAIGPT, HFApiModel

prompt = """You are a mental health expert tasked with classifying multiple-choice questions (MCQs) based on their difficulty for other mental health professionals.  Classify each given MCQ into one of the following categories: Easy, Medium, or Hard.

Classification Criteria:
* Easy:
    * Fact-based recall.
    * Single-domain, minimal reasoning.
    * Clear and distinct answer choices.
    * Limited jargon (if present, it is well-known in the field).

* Medium:
    * Requires reasoning over multiple facts.
    * May involve diagnostic criteria (e.g., DSM-5, ICD-11).
    * Moderate medical jargon (assumes familiarity with standard terminology).
    * Options may be nuanced (more than one option could seem correct at first glance).

* Hard:
    * Requires synthesis across multiple domains (biological, psychological, and social factors).
    * Complex medical terminology and nuanced answer choices (options may contain closely related terms requiring expert differentiation).
    * Multihop reasoning (the answer is not directly stated in the question; experts must infer it by applying knowledge from different areas).
    * May involve differential diagnosis or treatment trade-offs (more than one answer might seem plausible, but only one is fully correct).

Input Format:

ID: <Unique Question ID>
Question: <The question text>
Options:
Option1: <Option 1 text>
Option2: <Option 2 text>
Option3: <Option 3 text>
Option4: <Option 4 text>

The output should be in the following format for each question:

ID: <Unique Question ID>
Question: <The Question text in the output should be exactly the same as the input.>
Difficulty: <Easy, Medium or Hard>

Learn from this examples:
Questions:
ID: 1
Question: Which neurotransmitter is primarily associated with mood regulation?
Options:
Option1: Dopamine
Option2: Serotonin
Option3: Glutamate
Option4: GABA

ID: 2
Question: A patient with bipolar disorder is experiencing a mixed episode with agitation and psychotic features. Which pharmacological approach is most appropriate for acute stabilization?
Options:
Option1: Lithium monotherapy
Option2: Antidepressant plus a mood stabilizer
Option3: Atypical antipsychotic plus a mood stabilizer
Option4: Benzodiazepines alone

Output:
ID: 1
Question: Which neurotransmitter is primarily associated with mood regulation?
Difficulty: Easy

ID: 2
Question: A patient with bipolar disorder is experiencing a mixed episode with agitation and psychotic features. Which pharmacological approach is most appropriate for acute stabilization?
Difficulty: Hard

Instructions to the LLM:  
- Carefully consider the complexity of the question, the required knowledge, and the nuances of the answer choices when classifying the difficulty.  
- Assume the perspective of a mental health expert.  
- Output should strictly adhere to the specified format.  
- Do not provide explanations beyond the difficulty level.  
- The Question text in the output should be exactly the same as the input.
- Maintain the order of the questions as they appear in the input.

Read the following questions carefully and assign difficulty level to each question:
Questions:
{questions}

Output:
"""

csv_lock = Lock()

def extract_ids_question_difficulty(s: str):
    ids = re.findall(r"[\W]*ID[\W]*: (\d+)", s)
    questions = re.findall(r"[\W]*Question[\W]*: (.*)", s)
    difficulties = re.findall(r"[\W]*Difficulty[\W]*: (.*)", s)
    print("\n\nDIFF = ", difficulties)
    return (ids, questions, difficulties)

def build_prompt(batch):
    questions = []
    mcq_template = "ID: {id}\nQuestion: {question}\nOptions:\nOption1: {option1}\nOption2: {option2}\nOption3: {option3}\nOption4: {option4}\n"
    for idx, (_, row) in enumerate(batch.iterrows()):
        question = row["question"]
        options = [row[f"option{i}"] for i in range(1, 5)]
        mcq = mcq_template.format(id=idx+1, question=question, option1=options[0], option2=options[1], option3=options[2], option4=options[3])
        questions.append(mcq)
    return prompt.format(questions="\n".join(questions))

def run_inference(model: BaseModel, df: pd.DataFrame):
    new_prompt = build_prompt(df)
    response = model.generate_text(new_prompt)
    print(response)
    ids, questions, difficulties = extract_ids_question_difficulty(response)
    difficulties = [d.strip() for d in difficulties]
    responses = [response] * len(difficulties)
    return list(zip_longest(difficulties, responses)), response

def evaluate(model, batch, reslog):
    outputs, response = run_inference(model, batch)
    print("Len of outputs:", len(outputs))
    rows = []
    for idx, (_, row) in enumerate(batch.iterrows()):
        if idx >= len(outputs):
            rows.append(list(row.values) + [None, response])
        else: 
            rows.append(list(row.values) + list(outputs[idx]))
    with csv_lock:
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

    result_logger = CsvLogger(experiment_folder_path, "test_results", header + ["difficulty", "response"])

    batch_size = 10

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            futures.append(executor.submit(evaluate, model, batch, result_logger))
        for future in tqdm(futures):
            future.result()



