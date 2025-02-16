import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
args = parser.parse_args()
df = pd.read_csv(args.input_csv)
df.drop(columns=['as_response', "preethi ma'am response_as",
       'mg_response', "preethi ma'am response_mg", 'abstract', 'snippet', 'justification', 'decision', 'response'], inplace=True)
df.to_csv(args.output_csv, index=False)