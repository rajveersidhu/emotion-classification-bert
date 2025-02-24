# train.py - Trains the BERT model

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def train_model():
    texts, labels = load_data("data/emotion_dataset.csv")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len=128)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained("models/emotion_model")

if __name__ == "__main__":
    train_model()
