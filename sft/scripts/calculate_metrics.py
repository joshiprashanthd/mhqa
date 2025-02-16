import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import argparse

def calculate_metrics(df):
    labels = [1, 2, 3, 4]
    
    y_true = df["correct_option_number"].tolist()
    y_pred = df["predictions"].tolist()

    # if y_true and pred differ in size print 
    if len(y_true) != len(y_pred):
        print(f"y_true and y_pred differ in size: {len(y_true)} != {len(y_pred)}")
        print("y_true:", y_true)
        print("y_pred:", y_pred)
        return

    print("Unique values in y_true (correct answers):", set(y_true))
    print("Unique values in y_pred (model predictions):", set(y_pred))

    # Ensure labels exist in y_true before computing metrics
    if not any(x in labels for x in y_true):
        raise ValueError(f"No valid labels found in correct_option column. Found values: {set(y_true)}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:",'[xlabel: Prediction, ylabel: Ground Truth]\n', cm)
    
    correct_predictions = np.trace(cm)  # Sum of diagonal elements
    total_predictions = np.sum(cm)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    accuracy2 = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("Accuracy2:", accuracy2)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.diag(cm) / np.sum(cm, axis=0)
        precision2 = precision_score(y_true, y_pred, average='macro')
        print("Precision:", np.mean(precision))
        print("Precision2:", precision2)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        recall2 = recall_score(y_true, y_pred, average='macro')
        f1 = 2 * (precision * recall) / (precision + recall)
        f12 = f1_score(y_true, y_pred, average='macro')

        # Handle NaNs
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        f1 = np.nan_to_num(f1)

        print('f1:', np.mean(f1))
        print('f12:', f12)

    correct_predictions2 = np.sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
    total_predictions2 = len(y_true)
    if correct_predictions2 != correct_predictions:
        # flag error
        print(f"Correct predictions differ!!!!!!!!!!!!!!!: {correct_predictions} != {correct_predictions2}")
    if total_predictions2 != total_predictions:
        # flag error
        print(f"Total predictions differ!!!!!!!!!!!!!!!: {total_predictions} != {total_predictions2}")
    return precision2, recall2, f12, accuracy2, correct_predictions2, total_predictions2

def save_df(df, filename):
    # first load the contents of the file, then append the new data and save it back
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for multiple models')
    parser.add_argument('--resultFileDirectory', type=str, required=True, help='Directory containing result files')
    parser.add_argument('--original_data', type=str, required=True, help='Path to the original data file')
    parser.add_argument('--final_results', type=str, required=True, help='Path to the final results file')

    args = parser.parse_args()

    resultFileDirectory = args.resultFileDirectory
    original_data = args.original_data
    final_results = args.final_results

    original = pd.read_csv(original_data)

    # delete file metrics_output.csv if it exists
    if os.path.exists(final_results):
        os.remove(final_results)

    for file in os.listdir(resultFileDirectory):
        # initialize final results dataframe with columns: model, precision, recall, f1, accuracy, total_rows
        df_results = pd.DataFrame(columns=['model', 'precision', 'recall', 'f1', 'accuracy', 'correct_rows', 'total_rows'])

        if file.endswith('.csv'):
            resultFile = os.path.join(resultFileDirectory, file)
            print('Processing Result file:', resultFile)      
            
            results = pd.read_csv(resultFile)
            print('Results shape:', results.shape)

            # Keep only results that exist in the original dataset
            original["question_cleaned"] = original["question"].str.strip().str.lower()
            results["question_cleaned"] = results["question"].str.strip().str.lower()
            valid_results = results[results["question_cleaned"].isin(original["question_cleaned"])].copy()
            
            print('Valid results shape:', valid_results.shape)

            # Drop rows where 'predictions' or 'correct_option' are NaN
            valid_results.dropna(subset=['predictions', 'correct_option_number'], inplace=True)
            print('Valid results after dropping NaNs:', valid_results.shape)

            # Convert predictions & correct_option to integers
            valid_results.loc[:, 'predictions'] = valid_results['predictions'].astype(float).astype(int)
            valid_results.loc[:, 'correct_option_number'] = valid_results['correct_option_number'].astype(int)

            # Filter only valid labels (1, 2, 3, 4)
            valid_labels = [1, 2, 3, 4]
            valid_results = valid_results[valid_results['correct_option_number'].isin(valid_labels)]
            valid_results = valid_results[valid_results['predictions'].isin(valid_labels)]
            print('Valid results after filtering valid labels:', valid_results.shape)

            print('Predictions dtype:', valid_results['predictions'].dtype)
            print('Correct option dtype:', valid_results['correct_option_number'].dtype)

            # Compute metrics
            overall_precision, overall_recall, overall_f1, overall_accuracy, correct_rows, total_rows = calculate_metrics(valid_results)

            print(f"Overall Precision: {overall_precision:.4f}")
            print(f"Overall Recall: {overall_recall:.4f}")
            print(f"Overall F1-Score: {overall_f1:.4f}")
            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            print(f"Total Correct Predictions: {total_rows}")
            print("Total Rows:", valid_results.shape[0])
            print("-" * 25)

            fileName = resultFile.split('/')[-1].split('.csv')[0]

            print('File Name:', fileName)
            # make dictionary for df_results
            metrics_dict = pd.DataFrame([{
                "model": f"{fileName}_overall",
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "accuracy": overall_accuracy,
                "correct_rows": correct_rows,
                "total_rows": total_rows
            }])
            df_results = pd.concat([df_results, metrics_dict], ignore_index=True)
            empty_row = pd.DataFrame([{col: "" for col in df_results.columns}])  # Empty row as a DataFrame
            df_results = pd.concat([df_results, empty_row], ignore_index=True)

            # calculate_topic_type_metrics(valid_results, "topic")
            topics = valid_results['topic'].unique()
            types = valid_results['type'].unique()
            print('Topics:', topics)
            print('Types:', types)
            
            for topic in topics:
                topic_results = valid_results[valid_results['topic'] == topic]
                print(f"Topic: {topic}")
                print("-" * 25)

                #make dictionary for df_results
                topic_precision, topic_recall, topic_f1, topic_accuracy, topic_correct_rows, topic_total_rows = calculate_metrics(topic_results)
                metrics_dict = pd.DataFrame([{
                    "model": f"{fileName}_topic_{topic}",
                    "precision": topic_precision,
                    "recall": topic_recall,
                    "f1": topic_f1,
                    "accuracy": topic_accuracy,
                    "correct_rows": topic_correct_rows,
                    "total_rows": topic_total_rows
                }])
                df_results = pd.concat([df_results, metrics_dict], ignore_index=True)
                
            df_results = pd.concat([df_results, empty_row], ignore_index=True)
            for type_ in types:
                type_results = valid_results[valid_results['type'] == type_]
                print(f"Type: {type_}")
                print("-" * 25)
                type_precision, type_recall, type_f1, type_accuracy, type_correct_rows, type_total_rows = calculate_metrics(type_results)
                metrics_dict = pd.DataFrame([{
                    "model": f"{fileName}_type_{type_}",
                    "precision": type_precision,
                    "recall": type_recall,
                    "f1": type_f1,
                    "accuracy": type_accuracy,
                    "correct_rows": type_correct_rows,
                    "total_rows": type_total_rows
                }])
                df_results = pd.concat([df_results, metrics_dict], ignore_index=True)

            print("-" * 50)
            df_results = pd.concat([df_results, empty_row], ignore_index=True)
            df_results = pd.concat([df_results, empty_row], ignore_index=True)

            save_df(df_results, final_results)

if __name__ == "__main__":
    main()
