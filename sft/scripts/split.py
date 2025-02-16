import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(csv_path, train_size=0.9, val_size=0.5, test_size=0.5, random_state=42):
    df = pd.read_csv(csv_path)
    
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df['correct_option_num'],
        random_state=random_state
    )
    
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df['correct_option_num'],
        random_state=random_state
    )
    
    # Save splits to CSV files
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "big_shuffled.csv"
    train_df, val_df, test_df = split_dataset(csv_path)