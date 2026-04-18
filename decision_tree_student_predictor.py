import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


CSV_PATH = "university_performance_habits_cleaned.csv"
TARGET_SOURCE_COLUMN = "final_exam_score"
TARGET_COLUMN = "predicted_performance"

PERFORMANCE_LABELS = ["Poor", "Average", "Good", "Very Good", "Excellent"]
PERFORMANCE_DEFINITIONS = {
    "Poor": "student did not meet the minimum requirements",
    "Average": "student met the basic requirements but was not outstanding",
    "Good": "student performed above average",
    "Very Good": "student performed strongly",
    "Excellent": "student achieved the highest performance",
}

# Human-friendly aliases for Mode A input parsing.
FEATURE_ALIASES = {
    "attendance": "class_attendance_percent",
    "class attendance": "class_attendance_percent",
    "attendance percent": "class_attendance_percent",
    "class attendance percent": "class_attendance_percent",
    "attedance": "class_attendance_percent",
    "assignment": "assignment_score",
    "assignment score": "assignment_score",
    "score assignment": "assignment_score",
    "course": "major",
    "relationship": "relationship_status",
    "relation status": "relationship_status",
    "status": "relationship_status",
    "job": "part_time_job",
    "part time": "part_time_job",
    "prep days": "exam_preparation_days",
    "preparation days": "exam_preparation_days",
    "exam prep": "exam_preparation_days",
    "exam preparation": "exam_preparation_days",
    "study hours": "study_hours_per_day",
    "gaming": "gaming_hours",
    "gaming time": "gaming_hours",
    "game hours": "gaming_hours",
    "online hours": "screen_time_hours",
    "screen time": "screen_time_hours",
    "stress": "mental_stress_level",
    "internet": "internet_quality",
}


@dataclass
class PredictorArtifacts:
    """Objects the CLI and Streamlit app reuse after one training run."""
    model: Pipeline
    feature_columns: List[str]
    reference_df: pd.DataFrame
    accuracy: float
    mse: float


def create_classification_target(df: pd.DataFrame, source_column: str) -> pd.DataFrame:
    """Create a balanced 5-level categorical target from exam scores using quantile bins."""
    df = df.copy()
    ranked_scores = df[source_column].rank(method="first")
    df[TARGET_COLUMN] = pd.qcut(
        ranked_scores,
        q=5,
        labels=PERFORMANCE_LABELS,
    )
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the cleaned CSV and create the target label used for classification."""
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    required = {TARGET_SOURCE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    return create_classification_target(df, TARGET_SOURCE_COLUMN)


def build_model_pipeline(feature_columns: List[str], X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and tree-classifier steps into one reusable pipeline."""
    numeric_features = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return model


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    drop_cols = {TARGET_COLUMN, TARGET_SOURCE_COLUMN, "student_id"}
    feature_columns = [col for col in df.columns if col not in drop_cols]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(df: pd.DataFrame) -> PredictorArtifacts:
    """Train the model once and package the artifacts needed for prediction screens."""
    X_train, X_test, y_train, y_test, feature_columns = split_data(df)
    model = build_model_pipeline(feature_columns, X_train)
    model.fit(X_train, y_train)

    # Accuracy measures exact class hits, while MSE shows how far off ordinal predictions are.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    label_to_score = {label: score for score, label in enumerate(PERFORMANCE_LABELS)}
    y_test_scores = y_test.map(label_to_score).astype(float)
    y_pred_scores = pd.Series(y_pred, index=y_test.index).map(label_to_score).astype(float)
    mse = mean_squared_error(y_test_scores, y_pred_scores)

    separator = "=" * 52
    print(f"\n{separator}")
    print("  MODEL TRAINING COMPLETE")
    print(separator)
    print(f"  Total dataset rows      : {len(X_train) + len(X_test):,}")
    print(f"  Training set size       : {len(X_train):,} rows (80%)")
    print(f"  Test set size           : {len(X_test):,} rows (20%)")
    print(f"  Number of features used : {len(feature_columns)}")
    print(separator)
    print("  CLASS DISTRIBUTION IN TEST SET")
    for label in PERFORMANCE_LABELS:
        count = int((y_test == label).sum())
        bar = "#" * (count // 20)
        print(f"  {label:<12}: {count:>4}  {bar}")
    print(separator)
    print("  EVALUATION METRICS")
    print(f"  Test Accuracy           : {accuracy:.3f}  ({accuracy * 100:.1f}%)")
    print(f"  Mean Squared Error      : {mse:.3f}")
    print(f"  Root MSE                : {mse ** 0.5:.3f}")
    print(separator)
    print("  CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, zero_division=0))

    return PredictorArtifacts(
        model=model,
        feature_columns=feature_columns,
        reference_df=X_train,
        accuracy=accuracy,
        mse=mse,
    )


def get_default_feature_values(
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Dict[str, object]:
    """Provide fallback values so incomplete user input can still be scored."""
    defaults: Dict[str, object] = {}
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            defaults[col] = reference_df[col].median()
        else:
            defaults[col] = reference_df[col].mode(dropna=True).iloc[0]
    return defaults


def parse_question_to_feature_values(question: str, feature_columns: List[str]) -> Dict[str, str]:
    """
    Parse user question containing feature-value pairs.
    Supported patterns (case-insensitive):
    - feature=value
    - feature is value
    - feature: value
    Example:
      age=21, gender=female, study_hours_per_day=4.5, part_time_job=yes
    """
    extracted: Dict[str, str] = {}
    feature_lookup = {col.lower(): col for col in feature_columns}

    # Build alias lookup that also accepts space-based keys like "assignment score".
    alias_lookup: Dict[str, str] = {}
    for col in feature_columns:
        alias_lookup[col.lower().replace("_", " ")] = col
    for alias, canonical in FEATURE_ALIASES.items():
        if canonical in feature_columns:
            alias_lookup[alias.lower().strip()] = canonical

    chunks = re.split(r"[,;]", question)
    for chunk in chunks:
        match = re.match(r"\s*([a-zA-Z_\s]+?)\s*(?:=|:|is)\s*(.+?)\s*$", chunk)
        if not match:
            continue

        raw_key, raw_value = match.group(1), match.group(2)
        normalized_key = raw_key.strip().lower().replace("_", " ")
        normalized_key = re.sub(r"\s+", " ", normalized_key)

        mapped_column = alias_lookup.get(normalized_key)
        if mapped_column is None:
            mapped_column = feature_lookup.get(normalized_key.replace(" ", "_"))

        if mapped_column:
            extracted[mapped_column] = raw_value.strip()

    return extracted


def parse_natural_language_feature_values(
    question: str,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Dict[str, object]:
    extracted: Dict[str, object] = {}
    lowered_question = question.lower()

    def set_if_available(column: str, value: object) -> None:
        if column in feature_columns:
            extracted[column] = value

    age_match = re.search(r"\bage\s*(?:is\s*)?(\d{1,2})\b", lowered_question)
    if age_match:
        set_if_available("age", age_match.group(1))

    online_match = re.search(
        r"(?:online|screen\s*time)(?:\s*for)?\s*(\d+(?:\.\d+)?)\s*hours?\s*(?:a|per)?\s*day",
        lowered_question,
    )
    if online_match:
        set_if_available("screen_time_hours", online_match.group(1))

    study_match = re.search(
        r"study(?:ing)?\s*(?:for)?\s*(\d+(?:\.\d+)?)\s*hours?\s*(?:a|per)?\s*day",
        lowered_question,
    )
    if study_match:
        set_if_available("study_hours_per_day", study_match.group(1))

    gaming_match = re.search(
        r"gaming(?:\s*time)?(?:\s*for)?\s*(\d+(?:\.\d+)?)\s*hours?\s*(?:a|per)?\s*day",
        lowered_question,
    )
    if gaming_match:
        set_if_available("gaming_hours", gaming_match.group(1))

    prep_match = re.search(
        r"(?:exam\s*)?preparation(?:\s*days?)?(?:\s*(?:is|of|for))?\s*(\d+(?:\.\d+)?)",
        lowered_question,
    )
    if prep_match:
        set_if_available("exam_preparation_days", prep_match.group(1))

    assignment_match = re.search(r"assignment\s*score\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", lowered_question)
    if assignment_match:
        set_if_available("assignment_score", assignment_match.group(1))

    attendance_match = re.search(r"attendance\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", lowered_question)
    if attendance_match:
        set_if_available("class_attendance_percent", attendance_match.group(1))

    if re.search(r"part\s*time\s*job|working\s+(?:on\s+)?part\s*time|has\s+a\s+part\s*time\s+job", lowered_question):
        set_if_available("part_time_job", "Yes")
    elif re.search(r"no\s+part\s*time\s+job|without\s+a\s+part\s*time\s+job", lowered_question):
        set_if_available("part_time_job", "No")

    if re.search(r"in\s+a\s+relationship|has\s+a\s+relationship|has\s+relationship", lowered_question):
        set_if_available("relationship_status", "In a Relationship")
    elif re.search(r"single", lowered_question):
        set_if_available("relationship_status", "Single")

    if re.search(r"low|less|poor", lowered_question) and re.search(r"attendance", lowered_question):
        set_if_available("class_attendance_percent", 45)
    if re.search(r"very\s+low|extremely\s+low", lowered_question) and re.search(r"attendance", lowered_question):
        set_if_available("class_attendance_percent", 15)

    if re.search(r"high|strong|good", lowered_question) and re.search(r"assignment", lowered_question):
        set_if_available("assignment_score", 85)
    if re.search(r"low|poor|weak", lowered_question) and re.search(r"assignment", lowered_question):
        set_if_available("assignment_score", 35)

    if re.search(r"high|too much|a lot", lowered_question) and re.search(r"gaming", lowered_question):
        set_if_available("gaming_hours", 5)
    if re.search(r"low|little", lowered_question) and re.search(r"gaming", lowered_question):
        set_if_available("gaming_hours", 1)

    if "computer science" in lowered_question:
        set_if_available("major", "Computer Science")

    # Match known categorical values from the training data directly in the sentence.
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(reference_df[column]):
            continue
        for value in reference_df[column].dropna().astype(str).unique():
            value_text = value.strip().lower()
            if value_text and value_text in lowered_question:
                extracted[column] = value

    return extracted


def extract_features_from_question(
    question: str,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Dict[str, object]:
    """Combine direct key=value parsing with lighter natural-language matching."""
    extracted = parse_question_to_feature_values(question, feature_columns)
    natural_language_extracted = parse_natural_language_feature_values(
        question,
        feature_columns,
        reference_df,
    )
    extracted.update(natural_language_extracted)
    return extracted


def convert_types_for_row(row: Dict[str, object], reference_df: pd.DataFrame) -> Dict[str, object]:
    converted = {}
    for col, value in row.items():
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            try:
                converted[col] = float(value)
            except (ValueError, TypeError):
                converted[col] = reference_df[col].median()
        else:
            converted[col] = str(value).strip().title()
    return converted


def build_single_input_from_question(
    question: str,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    extracted = extract_features_from_question(question, feature_columns, reference_df)

    # Fill missing fields with training-set defaults, then override using question values.
    default_row = get_default_feature_values(feature_columns, reference_df)
    default_row.update(extracted)
    converted = convert_types_for_row(default_row, reference_df)
    return pd.DataFrame([converted], columns=feature_columns)


def build_single_input_from_answers(
    answers: Dict[str, object],
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    default_row = get_default_feature_values(feature_columns, reference_df)
    default_row.update(answers)
    converted = convert_types_for_row(default_row, reference_df)
    return pd.DataFrame([converted], columns=feature_columns)


def apply_rule_based_override(row_df: pd.DataFrame, explicit_features: Set[str] | None = None) -> str | None:
    """Apply strict business rules before falling back to the learned model prediction."""
    row = row_df.iloc[0]

    attendance = float(row.get("class_attendance_percent", 100))
    assignment = float(row.get("assignment_score", 100))
    prep_days = float(row.get("exam_preparation_days", 30))
    age = float(row.get("age", 18))
    screen_time = float(row.get("screen_time_hours", 0))

    major = str(row.get("major", "")).strip().lower()
    part_time_job = str(row.get("part_time_job", "")).strip().lower()

    provided = explicit_features or set()

    def has_all(columns: Set[str]) -> bool:
        return not explicit_features or columns.issubset(provided)

    # Hard failure rule for extreme academic risk.
    if (
        ("class_attendance_percent" in provided and attendance <= 10)
        or ("assignment_score" in provided and assignment <= 20)
        or not explicit_features and (attendance <= 10 or assignment <= 20)
    ):
        return "Poor"

    # Hard failure rule for the combined risk pattern requested by the user.
    is_job_risk = part_time_job == "yes"
    is_course_risk = major == "computer science"
    is_age_risk = age >= 28
    is_academic_risk = attendance <= 50 and assignment <= 50 and prep_days <= 5

    if has_all({"class_attendance_percent", "assignment_score", "exam_preparation_days", "part_time_job", "major", "age"}) and is_academic_risk and is_job_risk and is_course_risk and is_age_risk:
        return "Poor"

    # Mixed-risk rule requested by the user: older student, high online time,
    # part-time job, low attendance, but high assignment results.
    is_average_risk_pattern = (
        age >= 28
        and screen_time >= 12
        and is_job_risk
        and attendance <= 60
        and assignment >= 75
    )
    if has_all({"age", "screen_time_hours", "part_time_job", "class_attendance_percent", "assignment_score"}) and is_average_risk_pattern:
        return "Average"

    high_attendance = attendance >= 95
    high_assignment = assignment >= 95
    strong_prep = prep_days >= 14
    low_screen_time = screen_time <= 4
    strong_gpa = float(row.get("gpa", 0)) >= 3.8
    strong_study = float(row.get("study_hours_per_day", 0)) >= 4

    if has_all({"class_attendance_percent", "assignment_score", "exam_preparation_days", "gpa"}) and high_attendance and high_assignment and strong_prep and strong_gpa and (low_screen_time or strong_study):
        return "Excellent"

    if has_all({"class_attendance_percent", "assignment_score", "exam_preparation_days"}) and attendance >= 90 and assignment >= 85 and prep_days >= 10:
        return "Very Good"

    return None


def build_prediction_explanation(
    row_df: pd.DataFrame,
    prediction: str,
    explicit_features: Set[str] | None = None,
) -> str:
    """Turn the chosen prediction into a human-readable explanation."""
    row = row_df.iloc[0]

    attendance = float(row.get("class_attendance_percent", 100))
    assignment = float(row.get("assignment_score", 100))
    prep_days = float(row.get("exam_preparation_days", 30))
    screen_time = float(row.get("screen_time_hours", 0))
    study_hours = float(row.get("study_hours_per_day", 0))
    gpa = float(row.get("gpa", 0))
    part_time_job = str(row.get("part_time_job", "")).strip().lower()
    relationship_status = str(row.get("relationship_status", "")).strip()

    provided = explicit_features or set(row_df.columns)

    def mentioned(column: str) -> bool:
        return explicit_features is None or column in provided

    risk_reasons: List[str] = []
    support_reasons: List[str] = []
    context_reasons: List[str] = []

    if mentioned("class_attendance_percent") and attendance <= 20:
        risk_reasons.append("very low class attendance")
    elif mentioned("class_attendance_percent") and attendance <= 60:
        risk_reasons.append("low class attendance")
    elif mentioned("class_attendance_percent") and attendance >= 85:
        support_reasons.append("high class attendance")

    if mentioned("assignment_score") and assignment <= 20:
        risk_reasons.append("very low assignment scores")
    elif mentioned("assignment_score") and assignment <= 50:
        risk_reasons.append("low assignment scores")
    elif mentioned("assignment_score") and assignment >= 75:
        support_reasons.append("strong assignment results")

    if mentioned("exam_preparation_days") and prep_days <= 5:
        risk_reasons.append("limited exam preparation time")
    elif mentioned("exam_preparation_days") and prep_days >= 12:
        support_reasons.append("solid exam preparation time")

    if mentioned("screen_time_hours") and screen_time >= 10:
        risk_reasons.append("high online or screen time")

    if mentioned("study_hours_per_day") and study_hours >= 4:
        support_reasons.append("steady study hours")
    elif mentioned("study_hours_per_day") and study_hours <= 1:
        risk_reasons.append("very low study time")

    if mentioned("gpa") and gpa >= 3.7:
        support_reasons.append("a strong GPA")
    elif mentioned("gpa") and gpa <= 2.5:
        risk_reasons.append("a low GPA")

    if mentioned("part_time_job") and part_time_job == "yes" and (
        (mentioned("class_attendance_percent") and attendance <= 60)
        or (mentioned("exam_preparation_days") and prep_days <= 5)
    ):
        risk_reasons.append("a part-time job that may reduce study time")

    if mentioned("relationship_status") and relationship_status:
        if relationship_status.lower() == "in a relationship":
            context_reasons.append("relationship commitments")
        else:
            context_reasons.append(f"relationship status ({relationship_status})")

    if prediction == "Poor":
        if risk_reasons:
            if context_reasons:
                return (
                    f"Because of {', '.join(risk_reasons)}, and considering {', '.join(context_reasons)}, "
                    "this student might perform poorly and may not meet the minimum requirements."
                )
            return f"Because of {', '.join(risk_reasons)}, this student might perform poorly and may not meet the minimum requirements."
        return "Based on the overall pattern in the provided inputs, this student might perform poorly."

    if prediction == "Average":
        if risk_reasons and support_reasons:
            sentence = (
                f"This student might perform at an average level because {', '.join(risk_reasons)} are pulling performance down, "
                f"while {', '.join(support_reasons)} are helping keep the result stable."
            )
            if context_reasons:
                sentence += f" The description also includes {', '.join(context_reasons)}."
            return sentence
        if risk_reasons:
            if context_reasons:
                return (
                    f"This student might perform at an average level mainly because of {', '.join(risk_reasons)}, "
                    f"with {', '.join(context_reasons)} also part of the overall picture."
                )
            return f"This student might perform at an average level mainly because of {', '.join(risk_reasons)}."
        if support_reasons:
            if context_reasons:
                return (
                    f"This student might perform at an average level, supported by {', '.join(support_reasons)}, "
                    f"while still accounting for {', '.join(context_reasons)} in the description."
                )
            return f"This student might perform at an average level, supported by {', '.join(support_reasons)}."
        return "Based on the provided inputs, this student might perform at an average level."

    if prediction == "Good":
        if risk_reasons and support_reasons:
            sentence = (
                f"This student might perform well because {', '.join(risk_reasons)} are pulling performance down, "
                f"while {', '.join(support_reasons)} are helping keep the result stable."
            )
            if context_reasons:
                sentence += f" The description also includes {', '.join(context_reasons)}."
            return sentence
        if risk_reasons:
            if context_reasons:
                return (
                    f"This student might perform at a good level mainly because of {', '.join(risk_reasons)}, "
                    f"with {', '.join(context_reasons)} also part of the overall picture."
                )
            return f"This student might perform at a good level mainly because of {', '.join(risk_reasons)}."
        if support_reasons:
            if context_reasons:
                return (
                    f"This student might perform well, supported by {', '.join(support_reasons)}, "
                    f"while still accounting for {', '.join(context_reasons)} in the description."
                )
            return f"This student might perform well, supported by {', '.join(support_reasons)}."
        return "Based on the provided inputs, this student might perform at a good level."

    if prediction == "Very Good":
        if support_reasons:
            base = f"This student might perform very well because of {', '.join(support_reasons)}."
        else:
            base = "This student might perform very well based on the overall input pattern."
        if context_reasons:
            base += f" The description also includes {', '.join(context_reasons)}."
        return base

    if support_reasons:
        if context_reasons:
            return (
                f"This student might perform excellently because of {', '.join(support_reasons)}, "
                f"while the description also includes {', '.join(context_reasons)}."
            )
        return f"This student might perform excellently because of {', '.join(support_reasons)}."
    return "Based on the overall input pattern, this student might perform excellently."


def format_prediction_summary(prediction: str) -> str:
    definition = PERFORMANCE_DEFINITIONS.get(prediction)
    if not definition:
        return prediction
    return f"{prediction}: {definition}"


def build_improvement_recommendations(row_df: pd.DataFrame, prediction: str) -> List[str]:
    _, guidance_items = build_student_guidance(row_df, prediction)
    return guidance_items


def build_student_guidance(row_df: pd.DataFrame, prediction: str) -> Tuple[bool, List[str]]:
    row = row_df.iloc[0]
    recommendations: List[str] = []

    attendance = float(row.get("class_attendance_percent", 100))
    assignment = float(row.get("assignment_score", 100))
    prep_days = float(row.get("exam_preparation_days", 30))
    screen_time = float(row.get("screen_time_hours", 0))
    study_hours = float(row.get("study_hours_per_day", 0))
    gaming_hours = float(row.get("gaming_hours", 0))
    sleep_hours = float(row.get("sleep_hours", 8))
    stress = float(row.get("mental_stress_level", 0))
    part_time_job = str(row.get("part_time_job", "")).strip().lower()

    if attendance < 75:
        recommendations.append("Raise class attendance to at least 80% by planning weekly attendance targets.")
    if assignment < 70:
        recommendations.append("Improve assignment scores by starting early and using a 2-step review before submission.")
    if prep_days < 10:
        recommendations.append("Increase exam preparation to 10-14 days with a daily revision schedule.")
    if study_hours < 3:
        recommendations.append("Increase focused study time to at least 3-4 hours per day using timed study blocks.")
    if screen_time > 8:
        recommendations.append("Reduce non-academic screen time by 2-3 hours and move that time to revision.")
    if gaming_hours > 6:
        recommendations.append("Reduce gaming time to below 3 hours per day and shift that time to focused revision.")
    if sleep_hours < 6.5:
        recommendations.append("Aim for 7-8 hours of sleep to improve memory and concentration.")
    if stress >= 7:
        recommendations.append("Use stress-management habits (exercise, breaks, counseling support) to stabilize performance.")
    if part_time_job == "yes" and (study_hours < 3 or prep_days < 10):
        recommendations.append("Rebalance part-time work hours during exam weeks to protect study and revision time.")

    has_weakness = len(recommendations) > 0
    if has_weakness:
        return True, recommendations

    compliments: List[str] = []
    compliments.append("Great work. The profile shows no major academic weakness right now.")

    if gaming_hours > 2.5:
        compliments.append("To do even better, reduce gaming time and convert at least 1 hour into revision practice.")

    if prediction == "Excellent":
        compliments.append("Keep the same pace and maintain this discipline to stay at the highest performance level.")
        compliments.append("For further growth, add advanced practice and peer teaching to lock in mastery.")
    elif prediction == "Very Good":
        compliments.append("Keep this pace. You are performing strongly and are close to excellent level.")
        compliments.append("To move up, increase exam-focused revision quality and consistency.")
    elif prediction == "Good":
        compliments.append("You are above average. Keep the same pace and protect your current routine.")
        compliments.append("To move to Very Good, deepen weekly revision and accuracy in assignments.")
    elif prediction == "Average":
        compliments.append("You meet the basic requirements. Keep consistency while improving one weak area each week.")
    else:
        compliments.append("Use a weekly study plan and track attendance, assignments, and revision progress closely.")

    return False, compliments


def get_recommendations_for_question(
    question: str,
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> List[str]:
    prediction, _, row_df = predict_question_with_reason(
        question,
        model,
        feature_columns,
        reference_df,
    )
    return build_improvement_recommendations(row_df, prediction)


def get_recommendations_for_answers(
    answers: Dict[str, object],
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> List[str]:
    prediction, _, row_df = predict_answers_with_reason(
        answers,
        model,
        feature_columns,
        reference_df,
    )
    return build_improvement_recommendations(row_df, prediction)


def predict_question_with_reason(
    question: str,
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Tuple[str, str, pd.DataFrame]:
    extracted = extract_features_from_question(question, feature_columns, reference_df)
    explicit_features = set(extracted.keys())
    row_df = build_single_input_from_question(question, feature_columns, reference_df)
    override = apply_rule_based_override(row_df, explicit_features)
    prediction = override if override is not None else str(model.predict(row_df)[0])
    explanation = build_prediction_explanation(row_df, prediction, explicit_features)
    return prediction, explanation, row_df


def predict_answers_with_reason(
    answers: Dict[str, object],
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Tuple[str, str, pd.DataFrame]:
    row_df = build_single_input_from_answers(answers, feature_columns, reference_df)
    override = apply_rule_based_override(row_df, set(answers.keys()))
    prediction = override if override is not None else str(model.predict(row_df)[0])
    explanation = build_prediction_explanation(row_df, prediction, set(answers.keys()))
    return prediction, explanation, row_df


def predict_from_user_question(
    question: str,
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> str:
    prediction, _, _ = predict_question_with_reason(
        question,
        model,
        feature_columns,
        reference_df,
    )
    return prediction


def predict_from_answers(
    answers: Dict[str, object],
    model: Pipeline,
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> str:
    prediction, _, _ = predict_answers_with_reason(
        answers,
        model,
        feature_columns,
        reference_df,
    )
    return prediction


def prompt_for_feature_value(feature_name: str, reference_df: pd.DataFrame) -> object:
    label = feature_name.replace("_", " ")
    if pd.api.types.is_numeric_dtype(reference_df[feature_name]):
        default_value = reference_df[feature_name].median()
        raw_value = input(f"Enter {label} [default: {default_value}]: ").strip()
        if not raw_value:
            return default_value
        try:
            return float(raw_value)
        except ValueError:
            print(f"Invalid number. Using default value {default_value}.")
            return default_value

    default_value = reference_df[feature_name].mode(dropna=True).iloc[0]
    raw_value = input(f"Enter {label} [default: {default_value}]: ").strip()
    return raw_value if raw_value else default_value


def collect_plain_english_inputs(
    feature_columns: List[str],
    reference_df: pd.DataFrame,
) -> Dict[str, object]:
    answers: Dict[str, object] = {}
    print("\nAnswer the following prompts. Press Enter to accept the suggested default.")
    for feature_name in feature_columns:
        answers[feature_name] = prompt_for_feature_value(feature_name, reference_df)
    return answers


def _print_prediction_response(prediction: str, explanation: str, row_df: pd.DataFrame, mapped_count: int) -> None:
    """Print the full chat-style response: prediction, reason, detected fields, and guidance."""
    has_weakness, guidance_items = build_student_guidance(row_df, prediction)
    guidance_title = "Improvement plan" if has_weakness else "Keep the pace"
    separator = "-" * 52
    print(f"\n{separator}")
    print(f"  Prediction  : {format_prediction_summary(prediction)}")
    print(f"{separator}")
    print(f"  Reason      : {explanation}")
    print(f"  Fields used : {mapped_count} direct field(s) detected")
    print(f"{separator}")
    print(f"  {guidance_title}:")
    for item in guidance_items:
        print(f"    - {item}")
    print(separator)


def run_mode_a(artifacts: PredictorArtifacts) -> None:
    print("\nMode A: type a description or feature=value pairs.")
    print("Examples:")
    print("  age=21, attendance=85, assignment score=78, study hours=4")
    print("  A student who studies 3 hours a day with low attendance and a part-time job")
    user_question = input("\nDescribe the student: ").strip()
    prediction, explanation, row_df = predict_question_with_reason(
        user_question,
        artifacts.model,
        artifacts.feature_columns,
        artifacts.reference_df,
    )
    mapped_pairs = parse_question_to_feature_values(user_question, artifacts.feature_columns)
    _print_prediction_response(prediction, explanation, row_df, len(mapped_pairs))


def run_mode_b(artifacts: PredictorArtifacts) -> None:
    print("\nMode B: answer each prompt to describe the student.")
    answers = collect_plain_english_inputs(
        artifacts.feature_columns,
        artifacts.reference_df,
    )
    prediction, explanation, row_df = predict_answers_with_reason(
        answers,
        artifacts.model,
        artifacts.feature_columns,
        artifacts.reference_df,
    )
    _print_prediction_response(prediction, explanation, row_df, len(answers))


def load_and_train_predictor(csv_path: str = CSV_PATH) -> PredictorArtifacts:
    """Single entry point used by both the CLI flow and the Streamlit app."""
    df = load_dataset(csv_path)
    return train_model(df)


def main() -> None:
    # Running this file directly starts the command-line prediction workflow.
    print("Loading cleaned dataset...")
    print("Splitting data and training DecisionTreeClassifier...")
    artifacts = load_and_train_predictor(CSV_PATH)

    while True:
        print("\nChoose input mode:")
        print("  A - Describe student in one line (natural language or field=value)")
        print("  B - Answer guided prompts one by one")
        print("  Q - Quit")
        selected_mode = input("\nSelect Mode (A / B / Q): ").strip().upper()

        if selected_mode == "Q":
            print("Goodbye.")
            break
        elif selected_mode == "B":
            run_mode_b(artifacts)
        else:
            run_mode_a(artifacts)


if __name__ == "__main__":
    main()
