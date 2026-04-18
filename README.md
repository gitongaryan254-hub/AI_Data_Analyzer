# AI_Data_Analyzer

AI_Data_Analyzer is a student-performance analysis and prediction project built with Python, scikit-learn, and Streamlit.

The project workflow is:
1. Clean and profile the dataset.
2. Train a Decision Tree classifier.
3. Predict performance labels from user input in CLI or Streamlit.

Predicted labels:
- Poor
- Average
- Good
- Very Good
- Excellent

## Current repository structure

```text
AI_DATA_ANLYZER/
├─ decision_tree_student_predictor.py
├─ global_university_students_performance_habits.csv
├─ performance_visualization.png
├─ README.md
├─ requirements.txt
├─ streamlit_student_predictor.py
├─ university_performance_habits_cleaned.csv
├─ university_performance_habits_cleaning.py
└─ university_performance_habits_report.html
```

## Tech stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- sweetviz
- streamlit

## Setup (recommended)

Use one interpreter only for this repository: `.venv`.

Windows PowerShell:

```powershell
cd "C:\path\to\AI DATA ANLYZER"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

The last command should print a path ending in `.venv\Scripts\python.exe`.

## Run commands

From an activated `.venv` terminal:

```powershell
python university_performance_habits_cleaning.py
python decision_tree_student_predictor.py
python -m streamlit run streamlit_student_predictor.py
```

## Cross-machine reliability notes

- Always activate `.venv` before running any script.
- Do not run files with another global interpreter path (for example a separate uv Python), because this causes `ModuleNotFoundError` even if requirements are installed in `.venv`.
- Keep `requirements.txt` committed and updated after dependency changes.
- Keep dataset and scripts in the repository root as currently structured.

## What each script does

- `university_performance_habits_cleaning.py`
  - Loads raw CSV.
  - Cleans column names and duplicates.
  - Fills missing values deterministically.
  - Generates cleaned CSV, HTML report, and PNG visualization.

- `decision_tree_student_predictor.py`
  - Loads cleaned data.
  - Builds target labels.
  - Trains Decision Tree pipeline.
  - Reports test accuracy and MSE.
  - Supports CLI prediction flow.

- `streamlit_student_predictor.py`
  - Loads/trains predictor once with cache.
  - Supports NLP chat mode and guided form mode.
  - Shows prediction and explanation in UI.

## Troubleshooting

If you see `No module named pandas` (or similar):

1. Check interpreter path:

```powershell
python -c "import sys; print(sys.executable)"
```

2. If it is not `.venv\Scripts\python.exe`, activate `.venv` and try again.

3. Reinstall dependencies in the active environment:

```powershell
python -m pip install -r requirements.txt
```

## Contributing

Issues and pull requests are welcome.
