import pandas as pd

# Define the file path
file_path ="C:/Users/urbi1/Downloads/Thesis_repo_implementation/3_Implementation/2_Data_analysis/patents/pipeline/knowledge_abstract/output/causal_graph/filtered_final/filtered_complete_patent_causal_chains_pos_deaggregated.csv"

# Load the dataset
print(f"Loading file from: {file_path}")
df = pd.read_csv(file_path)

# Count occurrences of 'forward' and 'backward' in the 'category' column
forward_count = (df['category'] == 'forward').sum()
backward_count = (df['category'] == 'backward').sum()
mixed_count = (df['category'] == 'mixed').sum()

# Print the results
print("\n" + "=" * 40)
print("CAUSAL CHAIN CATEGORY COUNTS")
print("=" * 40)
print(f"Total rows with category 'forward'  : {forward_count:,}")
print(f"Total rows with category 'backward' : {backward_count:,}")
print(f"Total rows with category 'mixed'    : {mixed_count:,}")
print("=" * 40)