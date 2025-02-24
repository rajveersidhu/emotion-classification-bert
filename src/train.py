# train.py - Trains the BERT model

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist(), df['label'].tolist()

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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
    tokenizer.save_pretrained("models/emotion_model")

if __name__ == "__main__":
    train_model()
