"""
Terminal chat loop — replaces the old Streamlit web app.

Run with:
    .venv\\Scripts\\python.exe streamlit_student_predictor.py

Responds in the same style as the old chat app:
  Prediction, Reason, Detected fields count, Improvement plan / Keep the pace.
"""

from decision_tree_student_predictor import (
    CSV_PATH,
    build_student_guidance,
    format_prediction_summary,
    load_and_train_predictor,
    parse_question_to_feature_values,
    predict_question_with_reason,
)


def _print_response(prediction: str, explanation: str, row_df, mapped_count: int) -> None:
    """Print the chat-style prediction response to the terminal."""
    has_weakness, guidance_items = build_student_guidance(row_df, prediction)
    guidance_title = "Improvement plan" if has_weakness else "Keep the pace"
    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  Prediction  : {format_prediction_summary(prediction)}")
    print(sep)
    print(f"  Reason      : {explanation}")
    print(f"  Fields used : {mapped_count} direct field(s) detected")
    print(sep)
    print(f"  {guidance_title}:")
    for item in guidance_items:
        print(f"    - {item}")
    print(sep)


def main() -> None:
    """Load the model then run a continuous chat loop until the user quits."""
    print("Loading cleaned dataset and training model...")
    artifacts = load_and_train_predictor(CSV_PATH)

    print("\nExamples of what you can type:")
    print("  age=21, attendance=85, assignment score=78, study hours=4")
    print("  A student who studies 3 hours a day with low attendance and a part-time job")
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Describe the student: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        prediction, explanation, row_df = predict_question_with_reason(
            user_input,
            artifacts.model,
            artifacts.feature_columns,
            artifacts.reference_df,
        )
        mapped_pairs = parse_question_to_feature_values(user_input, artifacts.feature_columns)
        _print_response(prediction, explanation, row_df, len(mapped_pairs))


if __name__ == "__main__":
    main()
