import torch


def predict_text(model, text, vectorizer, device):
    sequence = vectorizer.transform([text])
    sequence = torch.tensor(sequence.toarray()).long().to(device)
    output = model(sequence)
    return torch.sigmoid(output)
