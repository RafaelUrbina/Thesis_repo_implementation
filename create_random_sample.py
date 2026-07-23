import pandas as pd
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere in the project
REPO_ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT_PATH))
from Utils.paths import REPO_ROOT

def create_random_sample_csv(input_csv_path: str, output_csv_path: str, num_rows: int, random_state: int = 42):
    """
    Creates a new CSV with a random sample of rows from an input CSV.

    This function reads a CSV file, selects a specified number of random rows,
    concatenates the 'Title' and 'Abstract' columns, and saves a new CSV
    with 'Lens ID', the concatenated column, and an empty 'precision' column.

    Args:
        input_csv_path (str): The full path to the input CSV file.
        output_csv_path (str): The full path where the output CSV will be saved.
        num_rows (int): The number of random rows to select.
        random_state (int): The seed for the random number generator. Defaults to 42.
    """
    try:
        # 1. Read the source CSV file into a pandas DataFrame
        print(f"Reading data from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")

        # Check if the number of requested rows is more than available
        if num_rows > len(df):
            print(f"Warning: Requested {num_rows} rows, but only {len(df)} are available. "
                  f"Using all {len(df)} rows instead.")
            num_rows = len(df)

        # 2. Get a random sample of 'x' rows
        print(f"Selecting {num_rows} random rows...")
        random_sample_df = df.sample(n=num_rows, random_state=RANDOM_STATE).copy()

        # 3. Concatenate 'Title' and 'Abstract' columns
        # Using .fillna('') ensures that if one of the columns is empty, it doesn't result in an empty concatenated string.
        print("Concatenating 'Title' and 'Abstract' columns...")
        random_sample_df['concatenated_column'] = random_sample_df['Title'].fillna('') + ' ' + random_sample_df['Abstract'].fillna('')

        # 4. Create the new empty 'precision' column
        random_sample_df['precision'] = ''

        # 5. Select and reorder the final columns
        final_df = random_sample_df[['Lens ID', 'concatenated_column', 'precision']]

        # 6. Save the new DataFrame to a new CSV file
        print(f"Saving the new CSV to '{output_csv_path}'...")
        final_df.to_csv(output_csv_path, index=False)
        print("New CSV file created successfully!")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_csv_path}'")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV: {e}. "
              "Please ensure 'Lens ID', 'Title', and 'Abstract' columns exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # Define the number of random rows you want to select
    NUMBER_OF_RANDOM_ROWS = 356  # You can change this value to whatever you need
    RANDOM_STATE = 76


    # Define the path to your input and output files
    # Using os.path.join for better cross-platform compatibility
    input_file = REPO_ROOT / 'precision-analysis.csv'
    output_file = REPO_ROOT / 'random_precision_sample.csv'

    # --- Run the function ---
    create_random_sample_csv(
        input_csv_path=input_file,
        output_csv_path=output_file,
        num_rows=NUMBER_OF_RANDOM_ROWS,
        random_state=RANDOM_STATE
    )
