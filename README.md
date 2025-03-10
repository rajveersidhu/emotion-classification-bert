# AI-Based Emotion Classification Using BERT

## Overview

This project implements a **BERT-based emotion classification model** to analyze and classify text into four emotions: **Happy, Sad, Angry, and Neutral**. The model leverages **transformer-based deep learning** to detect emotions from textual input.

## Features

- Fine-tuned **BERT** model for emotion detection.
- Trained using **PyTorch & Hugging Face Transformers**.
- Processes text input and classifies it into **four emotion categories**.
- Supports **real-time emotion prediction**.

## Dataset

- Uses a labeled dataset (`emotion_dataset.csv`) with text samples and emotion labels.
- The dataset is preprocessed using **tokenization and padding** before training.

## Model Architecture

- **BERT-base-uncased** from Hugging Face as the backbone model.
- **Fine-tuned classification head** for emotion classification.
- Optimized using **Adam optimizer & cross-entropy loss**.

## Installation

Clone the repository and install dependencies:

```bash
# Clone the repo
git clone https://github.com/rajveersidhu/emotion-classification-bert.git
cd emotion-classification-bert

# Install dependencies
pip install -r requirements.txt
```

## Training the Model

To train the BERT model, run:

```bash
python src/train.py
```

## Evaluating the Model

To evaluate the trained model on validation data:

```bash
python src/evaluate.py
```

## Real-Time Emotion Classification

To classify text in real-time:

```bash
python src/predict.py "I am feeling amazing today!"
```

**Output:** `Emotion: Happy`

## Dependencies

Install required libraries:

```bash
pip install torch transformers scikit-learn pandas numpy
```

## Folder Structure

```
├── data/
│   ├── emotion_dataset.csv
├── models/
│   ├── emotion_model/
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
├── notebooks/
│   ├── emotion_classification.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Enhancements

- Deploy model using **Flask or FastAPI**.
- Train with a **larger emotion dataset**.
- Improve model performance using **hyperparameter tuning**.

## Contributors

- **Rajveer Sidhu** ([GitHub](https://github.com/rajveersidhu))

## License

This project is licensed under the **MIT License**.

