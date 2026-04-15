 # 🧠 AI_Data_Analyzer

AI_Data_Analyzer is an end‑to‑end machine learning project that takes raw student academic data and turns it into **actionable performance insights**.  
Built by [`@gitongaryan254-hub`](https://github.com/gitongaryan254-hub), 
this project analyzes university student performance and habits, cleans messy CSVs, and predicts performance categories:

> **Poor · Average · Good · Very Good · Excellent**

The stack combines:
- **Pandas & NumPy** for data cleaning, statistics, and feature engineering  
- **Streetviz** Provides interactive profiling and summary reports of the dataset.

- **Seaborn** Generates statistical plots such as histograms, scatter plots, and correlation heatmaps.
- Together, they deliver clear insights into student performance patterns and complement the Decision Tree model built with scikit‑learn.
- **scikit-learn** with a **Decision Tree** model for interpretable predictions  
- **Matplotlib/Seaborn** for visualizations (exported to PNG)  
- **Streamlit** for an interactive web app that supports:
  - Structured feature–value input (sliders, dropdowns)
  - Guided **plain‑English prompts** that are mapped to model features

---

## 🔍 Project goals

- Build a **reproducible ML pipeline** for student performance prediction  
- Show the **full data science workflow**: raw data → cleaning → EDA → modeling → app  
- Provide a **teaching‑friendly** codebase with comments explaining AI logic and reasoning  
- Deploy the app so anyone can experiment with student performance scenarios in the browser

---

## 🗂️ Repository structure

```text
AI_Data_Analyzer/
├─ data/
│  ├─ raw/                  # Original Kaggle datasets
│  └─ processed/            # Cleaned / transformed datasets
├─ models/
│  └─ decision_tree_student_performance.pkl   # Trained model
├─ notebooks/
│  └─ 01_eda_and_cleaning.ipynb              # EDA + cleaning experiments
├─ app.py                                     # Streamlit front-end
├─ decision_tree_student_predictor.py         # ML backend (training + inference)
├─ visualization.png                          # Example performance visualization
├─ university_performance_habits_report.html  # Streetviz and Seaborn profiling report
# Streetviz provides interactive dataset profiling and summary reports.
# Seaborn is used for statistical visualizations (histograms, scatter plots, heatmaps).

├─ CAT2_Report.pdf                            # Project documentation & reflection
├─ requirements.txt                           # Python dependencies
└─ README.md
```

This layout separates raw data, processed data, models, notebooks, and the app, following common data‑science best practices.[web:28]

---

## ⚙️ Tech stack

- **Language**: Python  
- **Data & stats**: Pandas, NumPy  
- **Machine learning**: scikit‑learn (DecisionTreeClassifier)  
- **Visualization**: Matplotlib, Seaborn  
- **EDA automation**: ydata‑profiling  
- **App framework**: Streamlit

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/gitongaryan254-hub/AI_Data_Analyzer.git
cd AI_Data_Analyzer

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧹 Data cleaning & profiling

1. Place your CSVs in `data/raw/`, for example:
   - `dataset.csv`
   - `global_university_students_performance_habits_10000.csv`
   - `student_data.csv`

2. Run the cleaning and training pipeline (simplified example):

```bash
python decision_tree_student_predictor.py
```

Behind the scenes, the pipeline:

- Standardizes column names (lowercase, underscores)  
- Removes duplicates and fully empty columns  
- Handles missing values (numeric → mean, categorical → mode)  
- Encodes the **performance_category** into ordered classes  
- Trains a **Decision Tree** model and saves it to `models/decision_tree_student_performance.pkl`  

3. Generate a profiling report (optional, can also be done from the app):

- A **ydata‑profiling** report is saved as  
  `university_performance_habits_report.html`,  
  giving a full EDA of the student performance dataset.[web:16][web:19]

---

## 🎛️ Running the Streamlit app

```bash
streamlit run app.py
```

Features inside the app:

- **Structured Feature Input**  
  - Numeric features exposed as sliders  
  - User describes a student via direct feature values  
  - Model outputs predicted performance category (encoded class)

- **Plain‑English Prompt Mode**  
  - User types text like:  
    *"The student studies a lot, rarely misses lectures, but has a part‑time job."*  
  - A simple rule‑based layer maps this description to feature assumptions  
  - The Decision Tree predicts the performance class based on those features

- **Profiling Tab**  
  - View or generate the `university_performance_habits_report.html` profiling report from within the app

---

## 🌐 Deployment notes

You can deploy this project to several platforms:

### Streamlit Community Cloud

1. Push this repository to GitHub under  
   `gitongaryan254-hub/AI_Data_Analyzer`
2. Go to Streamlit Cloud and create a new app
3. Point it to:
   - Repository: `AI_Data_Analyzer`
   - Branch: `main`
   - File: `app.py`
4. Ensure `requirements.txt` is present and up to date so Streamlit installs all dependencies.[web:13]

### Google Colab

- Clone the repo in a Colab notebook  
- Install dependencies with `pip install -r requirements.txt`  
- Use the notebook for:
  - Data cleaning experiments
  - Model training
  - Generating visualizations and saving `visualization.png`

### GitHub Pages

- GitHub Pages can host static files only  
- You can deploy artifacts like:
  - `university_performance_habits_report.html`
  - Additional static visualizations or documentation

---

## 🧪 Reproducibility & evaluation

- Random seeds are fixed in the training pipeline for reproducible runs  
- Train–test splits, accuracy, and classification reports are logged in the console  
- Visualizations (e.g., class distribution, feature importance) can be exported to `visualization.png` for reports and presentations.[web:17][web:20]

---

## 📝 Academic context

This project is designed to be **explainable** in an academic setting:

- Code comments explain:
  - Why a **Decision Tree** is used (interpretable set of rules)  
  - How categorical labels are encoded  
  - How data cleaning decisions affect model quality
- `CAT2_Report.pdf` documents:
  - Problem definition
  - Data preparation
  - Model design and evaluation
  - Reflection and future improvements

---

## 🚀 Roadmap / Future work

- Add more models (Random Forest, XGBoost) for comparison  
- Improve the plain‑English parser with NLP techniques  
- Add feature importance plots and partial dependence plots  
- Integrate authentication for saving user scenarios

---

## 🤝 Contributing

Pull requests and suggestions are welcome.  
If you have ideas for new visualizations, model improvements, or UI enhancements, feel free to open an issue or PR.

---

# a simple flowchart for this project🧐

this is a simple way to look on my project flow chart.
AI_Data_Analyzer/
├─ data/
│  ├─ raw/
│  │  ├─ dataset.csv
│  │  ├─ global_university_students_performance_habits_10000.csv
│  │  └─ student_data.csv
│  └─ processed/
│     └─ cleaned_student_data.csv
├─ models/
│  └─ decision_tree_student_performance.pkl
├─ notebooks/
│  └─ 01_eda_and_cleaning.ipynb
├─ app.py
├─ decision_tree_student_predictor.py
├─ visualization.png
├─ university_performance_habits_report.html
├─ CAT2_Report.pdf
├─ requirements.txt
└─ README.md

---

## 📫 Contact

For questions, feedback, or collaboration, reach out via GitHub issues.
Created by **[@gitongaryan254-hub](https://github.com/gitongaryan254-hub)**.  
you can also contact me via phone number : +254105185046
or also reach out to me on my instagram page @rayooh.tosh
am free for suggestions,improvements and collaborations for a better and more intelligent model.😇

---

## 😎 authers information
university : african international university
course : cybersecurity and AI
name : RYAN GITONGA
admn. no. : 251362DAI
email : gamingstosh@gmail.com
