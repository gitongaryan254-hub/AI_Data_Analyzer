import pandas as pd
import numpy as np
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Step 1: Load dataset
df = pd.read_csv("global_university_students_performance_habits.csv")

# Step 2: Inspect dataset
print("Shape:", df.shape)
print(df.info())
print(df.head())

# Step 3: Basic cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()

# Step 4: Handle missing values adaptively
for col in df.columns:
    if df[col].isnull().sum() > 0:
        unique_vals = df[col].dropna().unique()
        if df[col].dtype in ["int64", "float64"]:
            # Numeric → fill with median
            df[col] = df[col].fillna(df[col].median())
            print(f"Filled numeric column: {col} with median")
        else:
            # Categorical → adaptive strategy
            if len(unique_vals) == 0:
                # No values at all → default
                df[col] = df[col].fillna("Student did not give feedback")
                print(f"Filled categorical column: {col} with default label")
            elif len(unique_vals) <= 3:
                # Few values → reuse them explicitly
                df[col] = df[col].apply(
                    lambda x: x if pd.notnull(x) else random.choice(unique_vals)
                )
                print(f"Filled categorical column: {col} using its few existing values")
            else:
                # Many values → random fill from distribution
                df[col] = df[col].apply(
                    lambda x: x if pd.notnull(x) else random.choice(unique_vals)
                )
                print(f"Filled categorical column: {col} with random existing values")

print("Remaining missing values:\n", df.isnull().sum())

# Step 5: Save cleaned dataset
csv_path = r"C:\Users\HP\OneDrive\Desktop\AI DATA ANLYZER\university_performance_habits_cleaned.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved successfully at: {csv_path}")

# Step 6: Generate profiling report with Sweetviz
report = sv.analyze(df)
report_path = r"C:\Users\HP\OneDrive\Desktop\AI DATA ANLYZER\university_performance_habits_report.html"
report.show_html(report_path)
print(f"Sweetviz report generated at: {report_path}")
os.startfile(report_path)  # automatically open HTML report

# Step 7: NumPy statistics for key numeric features
print("Columns available:", df.columns)  # debug check
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

# Step 8: Visualization (auto-save PNG + display)
if "class_attendance_percent" in df.columns and "final_exam_score" in df.columns:
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

    png_path = r"C:\Users\HP\OneDrive\Desktop\AI DATA ANLYZER\performance_visualization.png"
    plt.savefig(png_path, dpi=150)

    print("Seaborn check -> Scatter plot created successfully.")
    print(f"Matplotlib check -> Plot saved as: {png_path}")

    # Display plot when script runs; saving happens automatically without user input.
    plt.show()
    plt.close()