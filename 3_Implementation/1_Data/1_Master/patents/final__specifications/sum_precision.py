import pandas as pd
import os
from Utils.paths import REPO_ROOT

def calculate_and_print_precision(csv_path: str):
    """
    Reads a CSV file, calculates the percentage of '1's and '0's in the
    'precision' column, and prints the results.

    Args:
        csv_path (str): The full path to the input CSV file.
    """
    try:
        # --- 1. Read the CSV file ---
        print(f"Reading data from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        print("Data loaded successfully.")

        # --- 2. Validate the 'precision' column ---
        if 'precision' not in df.columns:
            print("\nError: A 'precision' column was not found in the CSV file.")
            return

        # --- 3. Prepare the data for calculation ---
        # Drop rows where the 'precision' column is empty/null to avoid errors
        # and ensure we only calculate percentages based on actual entries.
        precision_series = df['precision'].dropna()

        if precision_series.empty:
            print("\nThe 'precision' column is empty. No percentages to calculate.")
            return

        # --- 4. Calculate percentages ---
        # value_counts(normalize=True) calculates the relative frequencies (percentages)
        # of unique values in the Series.
        percentages = precision_series.value_counts(normalize=True) * 100

        # Get the percentage for '1' and '0', defaulting to 0 if they don't exist
        percentage_of_ones = percentages.get(1, 0)
        percentage_of_zeros = percentages.get(0, 0)

        # --- 5. Print the results ---
        print("\n--- Precision Analysis Results ---")
        print(f"Total entries analyzed in 'precision' column: {len(precision_series)}")
        print(f"Percentage of '1's: {percentage_of_ones:.2f}%")
        print(f"Percentage of '0's: {percentage_of_zeros:.2f}%")
        print("---------------------------------")

    except FileNotFoundError:
        print(f"\nError: The file was not found at the specified path: '{csv_path}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # Define the path to your input CSV file
    FILE_PATH = REPO_ROOT / "processed_random_precision_sample.csv"

    # --- Run the analysis function ---
    calculate_and_print_precision(FILE_PATH)
