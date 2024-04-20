# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


class SequencesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]


def train_model(model, X_train, y_train, vectorizer, device):
    sequences_train = vectorizer.transform(X_train.tolist())
    train_dataset = SequencesDataset(sequences_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(20):
        epoch_loss = 0
        epoch_correct = 0
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.squeeze(1).long().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets.view(-1, 1))
            epoch_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))
            correct = (preds == targets.view(-1, 1)).float()
            epoch_correct += correct.sum().item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1} - Loss: {epoch_loss/len(train_loader):.5f} - Accuracy: {epoch_correct/len(train_dataset):.5f}')
