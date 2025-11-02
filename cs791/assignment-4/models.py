### Add necessary imports ###
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

### Complete the CNN class here ###
class CNN(nn.Module):
    def __init__(self, num_classes=10): # Add arguments and modify as needed
        super(CNN, self).__init__()
    def forward(self, x): # Modify as needed
        pass