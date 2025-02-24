# predict.py - Runs real-time emotion classification

def predict_emotion(text, model_path="models/emotion_model"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    emotion_classes = ["Happy", "Sad", "Angry", "Neutral"]
    return emotion_classes[torch.argmax(probs, dim=-1).item()]

if __name__ == "__main__":
    user_input = input("Enter text to classify emotion: ")
    print(f"Emotion: {predict_emotion(user_input)}")
