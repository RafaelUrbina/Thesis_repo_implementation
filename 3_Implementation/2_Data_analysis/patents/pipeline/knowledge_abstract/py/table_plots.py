import sys
from pathlib import Path
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# =====================================================================
# Setup project root imports & Path Logic
# =====================================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "supervised_link"
    / "output"
    / "causal_filtered_supervised_exploded.csv"
)

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "term_frequencies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Individual Output File Paths
TECH_BARCHART_PNG = OUTPUT_DIR / "top_technical_terms.png"
ATTR_BARCHART_PNG = OUTPUT_DIR / "top_attribute_terms.png"
FAIL_BARCHART_PNG = OUTPUT_DIR / "top_failure_terms.png"


# ==========================================
# 1. DATA LOADING & PARSING
# ==========================================

def load_and_preprocess_dataset(file_path: Path) -> pd.DataFrame:
    """Loads dataset and parses string-encoded list columns into Python lists."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    list_columns = [
        "occurrence_technical",
        "occurrence_failures",
        "occurrence_variable"  # attributes
    ]

    def safe_parse_list(val):
        if isinstance(val, list):
            return val
        if pd.isna(val) or not val:
            return []
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return []
            elif val:
                return [x.strip() for x in val.split(",") if x.strip()]
        return []

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    return df


# ==========================================
# 2. BAR CHART GENERATOR FUNCTION
# ==========================================

def plot_term_frequency_barchart(
    df: pd.DataFrame, 
    column_name: str, 
    title: str, 
    output_path: Path, 
    top_k: int = 20, 
    color_palette: str = "Blues_r"
):
    """
    Extracts, counts, and plots a horizontal bar chart of the Top K most frequent terms
    for a given column and saves it to disk.
    """
    # Flatten the list of terms across all rows
    all_terms = [term for sublist in df[column_name] if isinstance(sublist, list) for term in sublist]
    
    if not all_terms:
        print(f"No terms found in column '{column_name}' to plot.")
        return

    # Count term frequencies
    term_counts = Counter(all_terms).most_common(top_k)
    freq_df = pd.DataFrame(term_counts, columns=["term", "count"])

    # Plotting
    plt.figure(figsize=(10, max(5, top_k * 0.35)))
    
    ax = sns.barplot(
        data=freq_df, 
        x="count", 
        y="term", 
        palette=color_palette,
        edgecolor="black",
        linewidth=0.5
    )

    # Annotate bar values on top of each bar
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{int(width)}",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left", 
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

    plt.title(f"{title} (Top {top_k})", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Occurrence Count (Sentences)", fontsize=10)
    plt.ylabel("Term", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Extend x-limit slightly so annotations don't get clipped
    plt.xlim(0, max(freq_df["count"]) * 1.15)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved: {output_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 60)
    print("GENERATING SEPARATE TERM FREQUENCY BAR CHARTS")
    print("=" * 60)

    # 1. Load dataset
    print(f"\nLoading dataset from: {INPUT_CSV_PATH}")
    df = load_and_preprocess_dataset(INPUT_CSV_PATH)

    # 2. Chart 1: Technical Terms
    print("\nGenerating Technical Terms Frequency Chart...")
    plot_term_frequency_barchart(
        df,
        column_name="occurrence_technical",
        title="Most Frequent Technical Terms",
        output_path=TECH_BARCHART_PNG,
        top_k=20,
        color_palette="Blues_r"
    )

    # 3. Chart 2: Attribute Terms (occurrence_variable)
    print("Generating Attribute Terms Frequency Chart...")
    plot_term_frequency_barchart(
        df,
        column_name="occurrence_variable",
        title="Most Frequent Attribute / Variable Terms",
        output_path=ATTR_BARCHART_PNG,
        top_k=20,
        color_palette="Greens_r"
    )

    # 4. Chart 3: Failure Terms
    print("Generating Failure Terms Frequency Chart...")
    plot_term_frequency_barchart(
        df,
        column_name="occurrence_failures",
        title="Most Frequent Failure Terms",
        output_path=FAIL_BARCHART_PNG,
        top_k=20,
        color_palette="Reds_r"
    )

    print("\n" + "=" * 60)
    print(f"ALL 3 BAR CHARTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()