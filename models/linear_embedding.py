import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import embedding
from utils.cachedir import cache_dir

import numpy as np
import pickle

input_size = 768 * 43
output_size = 1

hidden_size = 5
batch_size = 4
num_epochs = 500
learning_rate = 0.00001

# don't use this yet


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearEmbeddingModel():
    def __init__(self, train_data, test_data):
        self.model = Net()
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = torch.nn.MSELoss()

    def train(self):
        opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        i = 0
        for _ in range(num_epochs):
            for batch in self.train_data:
                x, y = self.get_pair(batch)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % 1000 == 0:
                    print(loss.item())

                i += 1

    def test(self):
        self.model.eval()

        tolerance = 0.2
        correct = 0
        total = 0
        losses = []

        # In case of python < 3.5
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        with torch.no_grad():
            for batch in self.test_data:
                x, y = self.get_pair(batch)
                y_hat = self.model(x)
                losses.append(self.loss_fn(y_hat, y))

                for b in range(batch_size):
                    yb = y[b].item()
                    yhb = y_hat[b].item()
                    if isclose(yb, yhb, rel_tol=tolerance):
                        correct += 1
                    total += 1

        print(f'Test accuracy: {correct / total}')
        print(f'Test loss: {np.mean(losses)}')

    @staticmethod
    def get_pair(batch):
        (x1, x2, y) = batch
        x2 = torch.unsqueeze(x2, 1)
        x = torch.cat((x1, x2), 1)
        x = torch.flatten(x, 1)
        y = torch.Tensor(y)
        y = torch.unsqueeze(y, 1)
        return Variable(x), Variable(y)

    @staticmethod
    def load_data():
        train, test = embedding.create_dataset('data/train_funlines.csv')
        print('done splitting')
        pickle.dump(train, open(f'{cache_dir}/train.p', 'wb'))
        pickle.dump(test, open(f'{cache_dir}/test.p', 'wb'))
