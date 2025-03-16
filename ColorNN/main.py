import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_decision_boundary import plot_decision_boundary

# Definiujemy sieć neuronową
class ColorNN(nn.Module):
    def __init__(self):
        super(ColorNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # (x, y) → 128 neuronów
        self.fc2 = nn.Linear(128, 64) # 128 → 64 neuronów
        self.fc3 = nn.Linear(64, 2)   # 64 → 2 wyjścia (czerwony, niebieski)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Nie używamy softmax, bo CrossEntropyLoss go używa

# Wczytanie danych
df = pd.read_csv("flower_points.csv")  # Wczytujemy wcześniej zapisane punkty
X = torch.tensor(df[['X', 'Y']].values, dtype=torch.float32)  # Współrzędne
Y = torch.tensor([0 if c == 'blue' else 1 for c in df['Color']], dtype=torch.long)  # 0 = blue, 1 = red

plt.scatter(df['X'], df['Y'], c=df['Color'].tolist())  # Konwersja na listę
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Tworzymy model, funkcję straty i optymalizator
model = ColorNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Trening
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
        
# Końcowa dokładność
with torch.no_grad():
    final_outputs = model(X)
    final_predictions = torch.argmax(final_outputs, dim=1)
    final_accuracy = (final_predictions == Y).float().mean().item()
    print(f'Final Accuracy: {final_accuracy:.4f}')

# Rysowanie granicy dezycyjnej
plot_decision_boundary(lambda x: model(x), X, Y, colors=('red', 'blue'))
