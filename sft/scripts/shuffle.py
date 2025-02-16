import pandas as pd
import numpy as np
from tqdm import tqdm

def shuffle_options(df):
    # Create a copy of the dataframe
    shuffled_df = df.copy()
    
    # For each row
    for idx in tqdm(range(len(df))):
        # Get the options as a list
        options = [df.loc[idx, f'option{i}'] for i in range(1, 5)]
        correct_num = int(df.loc[idx, 'correct_option_num'].item())
        correct_answer = options[int(correct_num - 1)]
        
        # Shuffle the options
        np.random.shuffle(options)
        
        # Update the options in the shuffled dataframe
        for i in range(4):
            shuffled_df.loc[idx, f'option{i+1}'] = options[i]
        
        # Find the new position of the correct answer
        new_correct_num = options.index(correct_answer) + 1
        shuffled_df.loc[idx, 'correct_option_num'] = new_correct_num
    
    return shuffled_df

def main():
    # Read the input CSV file
    input_file = 'big_cleaned.csv'  # Change this to your input file path
    df = pd.read_csv(input_file)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle the options
    shuffled_df = shuffle_options(df)
    
    # Save the shuffled dataset
    output_file = 'big_shuffled.csv'  # Change this to your desired output path
    shuffled_df.to_csv(output_file, index=False)
    
    # Print statistics of correct_option_num distribution
    print("Distribution of correct options after shuffling:")
    print(shuffled_df['correct_option_num'].value_counts().sort_index())

if __name__ == "__main__":
    main()