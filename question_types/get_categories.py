import sys
sys.path.append(".")
import datetime

import csv
import pandas as pd
from model import OpenAIGPT
import re
from tqdm import tqdm

prompt = """For the following questions, identify their general type or category that defines the nature of the question. 

Here are the categories from which you have to choose from:
- What are the characteristics/features? 
- How to perform a procedure?
- What is the prevalence/incidence?
- Does a factor influence the output?
- What is the best method/approach?
- What is the effectiveness of a treatment?
- What is the relationship between variables?
- What is the cause/mechanism?
- None of the above

Output Format for each question:
ID: <question id>
Question: <question text>
Category: <category according to the question type>

Questions:
{questions}

Output:
"""

timestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model = OpenAIGPT("gpt-4o-mini", 'cuda')
output_writer = csv.writer(open(f"categories_{timestring}.csv", "w"))
output_writer.writerow(["question", "category", 'response'])

def extract_categories(response: str):
    matches = re.findall(r"Category: (.+)", response)
    return matches

def inference(batch):
    responses = []
    questions = []
    question_template = """
ID: {id}
Type: {type}
Question: {question}
"""
    for idx, (_, row) in enumerate(batch.iterrows()):
        questions.append(question_template.format(id=idx, type=row['type'], question=row['question']))
    questions_text = "\n".join(questions)
    new_prompt = prompt.format(questions=questions_text)
    response = model.generate_text(new_prompt)
    print(response)
    categories = extract_categories(response)
    responses = [response] * len(batch)
    return categories, responses

def run(batch):
    categories, responses = inference(batch)
    for idx, (_, row) in enumerate(batch.iterrows()):
        category = categories[idx]
        if idx >= len(categories):
            category = "N/A"
        output_writer.writerow([row['question'], category, responses[idx]])

df = pd.read_csv("../datasets/mhqa-b.csv")

samples = df.sample(100, random_state=42)
batch_size = 50
types = df.type.unique()

type_to_questions = {t: [] for t in types}

for type in types:
    typedf = df[df.type == type]
    type_to_questions[type] = typedf.sample(500, random_state=42)

all_questions = pd.concat([v for v in type_to_questions.values()])
shuffled_questions = all_questions.sample(frac=1, random_state=42)

for i in tqdm(range(0, len(shuffled_questions), batch_size)):
    batch = shuffled_questions[i:i+batch_size]
    run(batch)


