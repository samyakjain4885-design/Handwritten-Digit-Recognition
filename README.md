# Handwritten Number Recognition (1–100)

A local Streamlit application that recognizes handwritten numbers from 1 to 100. It uses a CNN trained on MNIST to classify individual digits, then combines the digit predictions into a whole number.

## Features

- Draw a number in the app with a mouse, or upload a PNG/JPG/BMP image.
- Preprocess the image and split digits using blank vertical gaps.
- Show the whole-number prediction, average confidence, and each digit's crop and confidence.
- Save actual-versus-predicted results to `test_results.csv`.
- Optionally save verified digit corrections in `custom_digit_samples/` for future training.
- Review incorrect predictions, compare drawing and upload results, and download the CSV.

## Run the app in VS Code on Windows

Open this folder in VS Code. If the `.venv` folder does not already exist, run this in its PowerShell terminal:

```powershell
py -3.12 -m venv .venv
```

Then install the project packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the digit model. The first run downloads MNIST and saves `mnist_digit_model.keras`. If verified correction samples exist, training includes them while keeping validation examples separate:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app in a browser. Draw or upload a number, then enter the actual number and select **Save test** to add a row to `test_results.csv`.

After setup and model training, you can also double-click `run_project.bat` in this folder to start the app.

## Project files

- `app.py` — image input, digit splitting, predictions, test history, and results dashboard.
- `train_model.py` — MNIST CNN training with small rotations and shifts for data augmentation.
- `requirements.txt` — Python packages used by the app.
- `run_project.bat` — one-click Windows launcher for the app.
- `mnist_digit_model.keras` — trained model created by `train_model.py`.
- `test_results.csv` — test history created when the first result is saved.
- `custom_digit_samples/` — optional labeled digit crops selected for future training.

## Evaluation notes

The latest training run reached **99.14% accuracy on the MNIST digit test split**. The app dashboard calculates whole-number accuracy from `test_results.csv`. That log can contain tests from multiple model versions and repeated trials, so use a separate fresh test set for a final evaluation and report its sample size and test method alongside the score.

## Limitations

- MNIST contains single digits; the app combines digits after segmenting them by empty columns.
- Write digits with a small gap. Touching or overlapping digits can be split incorrectly.
- The app is designed for numbers 1–100 and at most three digits.
- Confidence is a model score, not a guarantee that a prediction is correct.
- Only save a correction example for training after confirming the actual number. Keep final evaluation examples separate from training examples.
