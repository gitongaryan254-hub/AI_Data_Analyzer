import pandas as pd
import numpy as np
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
import os
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV_PATH = PROJECT_ROOT / "global_university_students_performance_habits.csv"
CLEANED_CSV_PATH = PROJECT_ROOT / "university_performance_habits_cleaned.csv"
REPORT_PATH = PROJECT_ROOT / "university_performance_habits_report.html"
PLOT_PATH = PROJECT_ROOT / "performance_visualization.png"


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values without introducing randomness into the dataset."""
    df = df.copy()

    # Preserve the meaning of missing AI tool names for students with zero AI usage.
    if {"favorite_ai_tool", "ai_tool_usage_hours"}.issubset(df.columns):
        no_usage_mask = (
            df["favorite_ai_tool"].isna()
            & df["ai_tool_usage_hours"].fillna(0).eq(0)
        )
        df.loc[no_usage_mask, "favorite_ai_tool"] = "No AI Tool Used"

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            if pd.isna(fill_value):
                fill_value = 0
            df[col] = df[col].fillna(fill_value)
            print(f"Filled numeric column: {col} with median ({fill_value})")
            continue

        non_null_values = df[col].dropna()
        if non_null_values.empty:
            fill_value = "Unknown"
        else:
            fill_value = non_null_values.mode().iloc[0]

        df[col] = df[col].fillna(fill_value)
        print(f"Filled categorical column: {col} with mode ({fill_value})")

    return df

def load_raw_dataset(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the raw dataset that feeds the rest of the cleaning workflow."""
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, remove duplicates, and resolve missing values."""
    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(" ", "_")
    cleaned_df = cleaned_df.drop_duplicates()
    return fill_missing_values(cleaned_df)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Show the dataset shape, schema, preview, and remaining missing values."""
    print("Shape:", df.shape)
    print(df.info())
    print(df.head())
    print("Remaining missing values:\n", df.isnull().sum())


def save_cleaned_dataset(df: pd.DataFrame, csv_path: Path = CLEANED_CSV_PATH) -> None:
    """Persist the cleaned CSV so the model and app can reuse the same data."""
    df.to_csv(csv_path, index=False)
    print(f"CSV saved successfully at: {csv_path}")


def generate_profile_report(df: pd.DataFrame, report_path: Path = REPORT_PATH) -> None:
    """Build the Sweetviz HTML report for quick visual exploration of the cleaned data."""
    report = sv.analyze(df)
    report.show_html(str(report_path))
    print(f"Sweetviz report generated at: {report_path}")
    try:
        webbrowser.open(report_path.resolve().as_uri())
    except Exception:
        # Keep the workflow resilient in headless or restricted environments.
        pass


def print_numeric_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the main performance-related numeric columns."""
    print("Columns available:", df.columns)
    key_numeric_features = [
        "study_hours_per_day",
        "class_attendance_percent",
        "assignment_score",
        "final_exam_score",
    ]

    print("\nNumPy statistics (mean, median, standard deviation):")
    for feature in key_numeric_features:
        if feature in df.columns:
            values = df[feature].dropna().to_numpy(dtype=float)
            print(
                f"- {feature}: "
                f"mean={np.mean(values):.2f}, "
                f"median={np.median(values):.2f}, "
                f"std={np.std(values):.2f}"
            )


def create_visualization(df: pd.DataFrame, plot_path: Path = PLOT_PATH) -> None:
    """Create and save the attendance-vs-score scatter plot used in the project outputs."""
    if "class_attendance_percent" not in df.columns or "final_exam_score" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="class_attendance_percent",
        y="final_exam_score",
        alpha=0.5,
    )
    plt.title("Class Attendance vs Final Exam Score")
    plt.xlabel("Class Attendance (%)")
    plt.ylabel("Final Exam Score")
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150)

    print("Seaborn check -> Scatter plot created successfully.")
    print(f"Matplotlib check -> Plot saved as: {plot_path}")
    plt.show()
    plt.close()


def main() -> None:
    # This is the full cleaning workflow executed when the script is run directly.
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_CSV_PATH}")

    raw_df = load_raw_dataset()
    cleaned_df = clean_dataset(raw_df)
    print_dataset_summary(cleaned_df)
    save_cleaned_dataset(cleaned_df)
    generate_profile_report(cleaned_df)
    print_numeric_statistics(cleaned_df)
    create_visualization(cleaned_df)


if __name__ == "__main__":
    main()