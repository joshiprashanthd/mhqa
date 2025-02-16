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

Learn from these examples to arrive at the right answer:

Example 1:
Question: What is the relationship between exercise addiction and body dysmorphic disorder in individuals without eating disorder symptomology?
Options:
Option1: Positive correlation
Option2: No relation 
Option3: Negative correlation
Option4: Unclear relationship

Justification: Positive correlation – Exercise addiction and body dysmorphic disorder (BDD) both involve an intense preoccupation with physical appearance, making a positive correlation likely.
Negative correlation – There is no strong evidence suggesting that higher exercise addiction reduces BDD symptoms, so this option is unlikely.
No relation – Given their shared focus on body image concerns, it is improbable that these conditions are unrelated.
Unclear relationship – While more research may refine the understanding, existing evidence suggests a clear positive correlation.
By elimination, the best answer is "Positive correlation," as both conditions often co-occur due to body image preoccupation.
Correct Option: 1

Example 2:
Question: Among adolescent users of hallucinogens, which of the following factors is significantly associated with higher rates of considering suicide?
Options: 
Option1: Cigarette use
Option2: Alcohol use
Option3: Feeling sad and hopeless
Option4: Physical exercise

Justification: Cigarette use – Linked to risky behaviors but not as directly to suicidal ideation as emotional distress.
Alcohol use – Associated with impulsivity, but not the strongest predictor of suicidal thoughts.
Physical exercise – Generally improves mental health and reduces suicide risk, making it the least likely factor.
Feeling sad and hopeless – A core symptom of depression, strongly correlated with suicidal ideation.
Thus, "Feeling sad and hopeless" is the most significant factor associated with considering suicide among adolescent hallucinogen users.
Correct Option: 3

Example 3:
Question: Which alternative indicator is suggested for assessing the severity of eating disorders beyond the DSM-5 classification?
Options: 
Option1: Drive for Thinness
Option2: Anxiety levels
Option3: BMI categories
Option4: Dietary restrictions

Justification:  Drive for Thinness – A well-recognized psychological measure directly linked to eating disorders, representing an intense preoccupation with weight loss. This makes it a strong severity indicator.
Anxiety levels – While anxiety is commonly associated with eating disorders, it is not a direct measure of their severity. It can be a contributing factor but not a defining indicator.
BMI categories – BMI is a physical measure used for diagnosis but does not capture the psychological severity of eating disorders. Many individuals with severe eating disorders may not have an extreme BMI.
Dietary restrictions – Although restrictive eating is a symptom, it varies widely among individuals and does not necessarily indicate severity. Some individuals may engage in restrictive eating without a severe disorder.
By elimination, Drive for Thinness is the best alternative indicator for assessing the severity of eating disorders.
Correct Option: 1

Example 4:
Question: What should targeted suicide prevention strategies focus on to address the link between childhood physical abuse and suicidal behaviors?
Options: 
Option1: Increased parental supervision
Option2: Reduction of youth aggression
Option3: Enhancing community support
Option4: Improving educational outcomes

Justification:  Increased parental supervision – While important, it doesn't directly address the emotional and behavioral impacts of abuse, such as aggression.
Reduction of youth aggression – Directly targets the emotional and behavioral consequences of childhood physical abuse, which can reduce the risk of suicidal behaviors.
Enhancing community support – Provides secondary support but doesn't directly address the root emotional issues related to childhood abuse.
Improving educational outcomes – Important for long-term development, but it doesn't directly address the immediate effects of abuse, like aggression and emotional distress.
By elimination, "Reduction of youth aggression" is the most focused and effective strategy to prevent suicidal behaviors linked to childhood abuse. 
Correct Option: 2

Output Format:
Correct Option: <1 or 2 or 3 or 4>
Justification: <justification for the answer>

Question: {question}
Options: 
Option1: {op1}
Option2: {op2}
Option3: {op3}
Option4: {op4}

Respond with Justification first then write the number corresponding to your choice (1 or 2 or 3 or 4) for Correct Option on the next line. 
Justification: """

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
        answer, justification = extract_answer_justification(generated_text)
        outputs.append([answer, justification, generated_text])
    return outputs

def evaluate(model, batch, reslog):
    outputs = run_inference(model, batch)
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
    parser.add_argument("--except_csv_path", help="path to the csv file to exclude", required=False)

    cmd_args = parser.parse_args()

    input_csv_path = Path(cmd_args.csv_path)
    log_folder_path = Path("./logs_few_shot")
    log_folder_path.mkdir(parents=True, exist_ok=True)
    sluggify = re.sub(r"[\.|\/|\-]", "_", cmd_args.csv_path)
    exp_name = Path(f"{cmd_args.model}_{sluggify}".replace("/","_"))
    experiment_folder_path = log_folder_path / exp_name
    experiment_folder_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cmd_args.model in openai_model_names:
        print('Using OPENAI')
        model = OpenAIGPT(cmd_args.model, device)
    else: 
        print("Using HuggingFace")
        model = HFApiModel(cmd_args.model, device)

    df = pd.read_csv(input_csv_path) 
    header = list(df.columns)

    if cmd_args.except_csv_path:
        except_df = pd.read_csv(cmd_args.except_csv_path)
        df = df[~df["question"].isin(except_df["question"])]

    result_logger = CsvLogger(experiment_folder_path, "test_results", header + ["predictions", "justification", "response"])

    batch_size = 10

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            futures.append(executor.submit(evaluate, model, batch, result_logger))
        for future in tqdm(futures):
            future.result()



