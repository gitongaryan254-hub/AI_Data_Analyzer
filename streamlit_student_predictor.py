import pandas as pd
import socket
import streamlit as st

from decision_tree_student_predictor import (
    CSV_PATH,
    PredictorArtifacts,
    build_student_guidance,
    format_prediction_summary,
    load_and_train_predictor,
    parse_question_to_feature_values,
    predict_answers_with_reason,
    predict_question_with_reason,
)


st.set_page_config(page_title="Student Performance Predictor", layout="centered")


@st.cache_resource
def get_predictor() -> PredictorArtifacts:
    return load_and_train_predictor(CSV_PATH)


def render_header() -> None:
    st.title("Student Performance Predictor")
    st.write(
        "Describe a student in natural language or answer guided prompts. "
        "The app uses the Decision Tree backend and responds with both a prediction and a reason."
    )

    local_ip = get_local_ip()
    st.info(
        "LAN mode only: use these links on the same Wi-Fi network.\n"
        f"- Laptop: http://localhost:8501\n"
        f"- Phone (same network): http://{local_ip}:8501"
    )
    st.caption("For same-network use only. Do not use public tunnels or router port forwarding.")


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def render_chat_mode(artifacts: PredictorArtifacts) -> None:
    st.subheader("Mode A: NLP chat")
    st.caption(
        "Examples: 'age=21, attendance=50, assignment score=80' or "
        "'A student age 30 staying online for 25 hours a day with low attendance and strong assignments.'"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Describe the student here...")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prediction, explanation, row_df = predict_question_with_reason(
        prompt,
        artifacts.model,
        artifacts.feature_columns,
        artifacts.reference_df,
    )
    has_weakness, guidance_items = build_student_guidance(row_df, prediction)
    mapped_pairs = parse_question_to_feature_values(prompt, artifacts.feature_columns)

    guidance_title = (
        "Improvement plan 📈"
        if has_weakness
        else "Keep the pace 🌟"
    )
    guidance_lines = "\n".join([f"- {item}" for item in guidance_items])
    guidance_block = f"\n\n{guidance_title}:\n{guidance_lines}"

    response = (
        f"Prediction 🎯: **{format_prediction_summary(prediction)}**\n\n"
        f"Reason 🧠: {explanation}\n\n"
        f"Detected direct fields 🔎: {len(mapped_pairs)}"
        f"{guidance_block}"
    )

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("Show interpreted values"):
            preview_columns = [
                "age",
                "study_hours_per_day",
                "screen_time_hours",
                "class_attendance_percent",
                "assignment_score",
                "exam_preparation_days",
                "part_time_job",
                "relationship_status",
                "major",
            ]
            visible_columns = [col for col in preview_columns if col in row_df.columns]
            st.dataframe(row_df[visible_columns], use_container_width=True)


def build_mode_b_inputs(artifacts: PredictorArtifacts) -> dict:
    answers = {}
    st.subheader("Mode B: Guided prompts")
    st.write("Answer each prompt and then click Predict.")

    columns = st.columns(2)
    for index, feature_name in enumerate(artifacts.feature_columns):
        reference_series = artifacts.reference_df[feature_name]
        label = feature_name.replace("_", " ").title()
        current_column = columns[index % 2]

        with current_column:
            if pd.api.types.is_numeric_dtype(reference_series):
                default_value = float(reference_series.median())
                answers[feature_name] = st.number_input(
                    label,
                    value=default_value,
                    key=f"mode_b_{feature_name}",
                )
            else:
                default_value = str(reference_series.mode(dropna=True).iloc[0])
                unique_values = sorted(str(value) for value in reference_series.dropna().unique())
                default_index = unique_values.index(default_value) if default_value in unique_values else 0
                answers[feature_name] = st.selectbox(
                    label,
                    options=unique_values,
                    index=default_index,
                    key=f"mode_b_{feature_name}",
                )

    return answers


def main() -> None:
    render_header()

    artifacts = get_predictor()
    st.caption(f"Backend model ready. Test accuracy: {artifacts.accuracy:.3f}")

    selected_mode = st.radio(
        "Choose input mode",
        options=["Mode A: NLP chat", "Mode B: guided prompts"],
        horizontal=True,
    )

    if selected_mode == "Mode A: NLP chat":
        render_chat_mode(artifacts)

    else:
        answers = build_mode_b_inputs(artifacts)
        if st.button("Predict from Mode B", type="primary"):
            prediction, explanation, row_df = predict_answers_with_reason(
                answers,
                artifacts.model,
                artifacts.feature_columns,
                artifacts.reference_df,
            )
            has_weakness, guidance_items = build_student_guidance(row_df, prediction)
            st.success(f"Prediction 🎯: {format_prediction_summary(prediction)}")
            st.info(f"Reason 🧠: {explanation}")
            if has_weakness:
                st.markdown("**Improvement plan 📈**")
            else:
                st.markdown("**Keep the pace 🌟**")
            for item in guidance_items:
                st.markdown(f"- {item}")
            with st.expander("Show interpreted values"):
                st.dataframe(row_df, use_container_width=True)


if __name__ == "__main__":
    main()
