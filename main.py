from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the iris dataset
class Model(nn.Module):
    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
model = Model()
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=12)
X_Train= torch.FloatTensor(X_Train)
X_Test= torch.FloatTensor(X_Test)
y_Train= torch.LongTensor(y_Train)
y_Test= torch.LongTensor(y_Test)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_Train)
    loss = criterion(y_pred, y_Train)
    losses.append(loss.detach().numpy())
    print(f'Epoch: {i} Loss: {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

