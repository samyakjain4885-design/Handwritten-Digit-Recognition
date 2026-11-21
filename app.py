"""Streamlit demo for recognizing separated handwritten digits from 1 to 100."""

from pathlib import Path
import csv
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


MODEL_PATH = Path(__file__).resolve().parent / "mnist_digit_model.keras"
RESULTS_PATH = Path(__file__).resolve().parent / "test_results.csv"
CUSTOM_SAMPLES_PATH = Path(__file__).resolve().parent / "custom_digit_samples"


@st.cache_resource
def load_model(model_timestamp: float):
    return tf.keras.models.load_model(MODEL_PATH)


def split_digit_image(image: Image.Image) -> list[np.ndarray]:
    """Split dark handwriting on a light background by blank vertical columns."""
    image = ImageOps.grayscale(image)
    image.thumbnail((900, 300))
    gray = np.asarray(image)
    ink = gray < 150
    if ink.mean() > 0.5:
        ink = ~ink

    # Ignore small specks by finding the overall ink bounds first.
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return []
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    active_cols = ink.any(axis=0)
    groups: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(active_cols):
        if active and start is None:
            start = i
        elif not active and start is not None:
            groups.append((start, i))
            start = None
    if start is not None:
        groups.append((start, len(active_cols)))

    digits = []
    for left, right in groups:
        crop = ink[:, left:right]
        cy, cx = np.where(crop)
        if len(cx) == 0:
            continue
        crop = crop[cy.min() : cy.max() + 1, cx.min() : cx.max() + 1]
        # Fit the digit into a 20x20 box, then center it on MNIST's 28x28 canvas.
        h, w = crop.shape
        scale = 20 / max(h, w)
        resized = Image.fromarray((crop * 255).astype("uint8")).resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
        )
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
        digits.append(np.asarray(canvas, dtype="float32") / 255.0)
    return digits


st.set_page_config(page_title="Handwritten Number Recognition", page_icon="✍️")
st.title("Handwritten Number Recognition")
st.write("Draw a number with your mouse or upload an image. The app recognizes numbers from 1 to 100.")
st.info("Keep a small gap between digits, for example: 5 8. Touching digits may not split correctly.")

if not MODEL_PATH.exists():
    st.warning("First train the MNIST model: run `python train_model.py` in the project terminal.")
    st.stop()

input_method = st.segmented_control(
    "Choose input method",
    options=["Draw here", "Upload an image"],
    default="Draw here",
    key="input_method",
)
image = None
input_name = ""

if input_method == "Draw here":
    if "canvas_reset" not in st.session_state:
        st.session_state.canvas_reset = 0
    if st.button("Clear drawing"):
        st.session_state.canvas_reset += 1
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=12,
        stroke_color="#111111",
        background_color="#FFFFFF",
        height=240,
        width=500,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key=f"number_canvas_{st.session_state.canvas_reset}",
    )
    if canvas_result.image_data is not None:
        canvas_pixels = np.asarray(canvas_result.image_data, dtype="uint8")
        if np.any(canvas_pixels[:, :, :3] < 245):
            image = Image.fromarray(canvas_pixels, mode="RGBA").convert("RGB")
            input_name = "mouse_drawing.png"
else:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        input_name = uploaded.name

current_prediction = None
current_confidence = None
if image is not None:
    caption = "Your drawing" if input_method == "Draw here" else "Uploaded handwriting"
    st.image(image, caption=caption, alt="Handwritten number submitted for recognition", width=350)
    digits = split_digit_image(image)
    if not digits:
        st.error("No handwriting found. Try a darker pen on a light background.")
    elif len(digits) > 3:
        st.error("Please upload one number with at most three digits.")
    else:
        model = load_model(MODEL_PATH.stat().st_mtime)
        probabilities = model.predict(np.stack(digits)[..., None], verbose=0)
        predictions = probabilities.argmax(axis=1)
        digit_confidences = probabilities.max(axis=1)
        number = int("".join(str(int(d)) for d in predictions))
        confidence = float(np.mean(digit_confidences))
        current_prediction = number
        current_confidence = confidence
        if 1 <= number <= 999:
            st.success(f"Predicted number: **{number}**")
        else:
            st.warning(f"Recognized digits: **{number}** (outside the 1–999 range)")
        st.metric("Average digit confidence", f"{confidence:.1%}")
        st.caption("Confidence is the model's score, not a guarantee that the answer is correct.")

        st.subheader("Digit-by-digit check")
        digit_columns = st.columns(len(digits))
        for index, (column, digit_image, digit_prediction, digit_confidence) in enumerate(
            zip(digit_columns, digits, predictions, digit_confidences), start=1
        ):
            with column:
                digit_preview = Image.fromarray((digit_image * 255).astype("uint8"))
                st.image(
                    digit_preview,
                    alt=f"Processed crop for digit {index}",
                    width=84,
                )
                st.write(f"Digit {index}: **{int(digit_prediction)}**")
                st.caption(f"Confidence: {digit_confidence:.1%}")

        st.subheader("Save this test result")
        with st.form("save_test_result"):
            actual_number = st.number_input(
                "What number did you write?",
                min_value=1,
                max_value=999,
                step=1,
                value=1,
            )
            use_for_training = st.checkbox(
                "Save these labeled digits as correction examples for future training",
                value=False,
                help="Select this only after checking that the actual number above is correct.",
            )
            submitted = st.form_submit_button("Save test")

        if submitted:
            result = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": input_name,
                "actual": int(actual_number),
                "predicted": current_prediction,
                "confidence_percent": round(current_confidence * 100, 1),
                "correct": current_prediction == int(actual_number),
            }
            new_file = not RESULTS_PATH.exists()
            with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=result.keys())
                if new_file:
                    writer.writeheader()
                writer.writerow(result)
            if use_for_training:
                actual_digits = str(int(actual_number))
                if len(actual_digits) == len(digits):
                    sample_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    for position, (label, digit_image) in enumerate(zip(actual_digits, digits), start=1):
                        label_folder = CUSTOM_SAMPLES_PATH / label
                        label_folder.mkdir(parents=True, exist_ok=True)
                        sample = Image.fromarray(
                            np.clip(digit_image * 255, 0, 255).astype("uint8")
                        )
                        sample.save(label_folder / f"{sample_time}_{position}.png")
                    st.success(
                        f"Test saved. {len(digits)} labeled digit example(s) added for future training."
                    )
                else:
                    st.warning(
                        "Test saved, but correction examples were not added: the detected digit count "
                        "does not match the number you entered."
                    )
            else:
                st.success("Test result saved.")

if RESULTS_PATH.exists():
    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    if history:
        correct_count = sum(row["correct"].lower() == "true" for row in history)
        test_count = len(history)
        incorrect_rows = [row for row in history if row["correct"].lower() != "true"]
        st.subheader("Test results summary")
        with st.container(horizontal=True):
            st.metric("Saved tests", test_count, border=True)
            st.metric("Correct predictions", correct_count, border=True)
            st.metric("Overall accuracy", f"{correct_count / test_count:.1%}", border=True)

        canvas_history = [row for row in history if row["image"] == "mouse_drawing.png"]
        upload_history = [row for row in history if row["image"] != "mouse_drawing.png"]
        st.subheader("Accuracy by input method")
        with st.container(horizontal=True):
            for label, rows in (
                ("Draw here", canvas_history),
                ("Upload image", upload_history),
            ):
                if rows:
                    method_correct = sum(row["correct"].lower() == "true" for row in rows)
                    method_value = f"{method_correct / len(rows):.1%} ({method_correct}/{len(rows)})"
                else:
                    method_value = "No saved tests"
                st.metric(f"{label} accuracy", method_value, border=True)

        st.download_button(
            "Download test results (CSV)",
            data=RESULTS_PATH.read_bytes(),
            file_name="handwritten_number_test_results.csv",
            mime="text/csv",
        )

        with st.expander(f"Review incorrect predictions ({len(incorrect_rows)})"):
            if incorrect_rows:
                st.dataframe(
                    incorrect_rows,
                    width="stretch",
                    hide_index=True,
                    alt="Saved tests where the predicted number did not match the written number",
                )
            else:
                st.info("No incorrect predictions have been saved yet.")

        st.subheader("All saved tests")
        st.dataframe(
            history,
            width="stretch",
            hide_index=True,
            alt="All saved handwritten number recognition tests",
        )
        st.caption(
            f"Full test history is saved in {RESULTS_PATH.name}. Overall accuracy includes every saved row."
        )
