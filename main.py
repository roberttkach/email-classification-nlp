import os
import torch
from data_utils import load_data, preprocess_data
from model import LSTM
from train import train_model
from predict import predict_text


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data = load_data()
X_train, y_train, X_test, y_test, vectorizer = preprocess_data(train_data, test_data)

model = LSTM(len(vectorizer.vocabulary_), 100, 64, 1).to(device)

train_model(model, X_train, y_train, vectorizer, device)

text = 'Get your discount code DC123456. To stop further messages, reply "stop". hh.ru. Customer Support 18001234567'
prediction = predict_text(model, text, vectorizer, device)
print(f"Prediction: {prediction.item()}")

os.makedirs('models', exist_ok=True)

torch.save(model.state_dict(), 'models/first_model.pth')
