import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import embedding
from utils.cachedir import cache_dir

import numpy as np
import pickle

input_size = 768
output_size = 1

hidden_size = 20
batch_size = 4
num_epochs = 500
learning_rate = 0.0001

# TODO this is a basic linear regression model, we can probably get
# higher accuracy with LSTM


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearModel():
    def __init__(self, train_data, test_data):
        self.model = Net()
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        i = 0
        for _ in range(num_epochs):
            for batch in self.train_data:
                x, y = self.get_pair(batch)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % (4000 / batch_size) == 0:
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
        x, y = batch
        y = torch.Tensor(y)
        y = torch.unsqueeze(y, 1)
        return Variable(x), Variable(y)

    @staticmethod
    def load_data():
        train, test = embedding.create_dataset('data/train_funlines.csv')
        print('done splitting')
        train = embedding.embed(train)
        test = embedding.embed(test)
        train = embedding.batch_embedded(train, batch_size)
        test = embedding.batch_embedded(test, batch_size)
        print('done embedding')
        pickle.dump(train, open(f'{cache_dir}/train_embedded.p', 'wb'))
        pickle.dump(test, open(f'{cache_dir}/test_embedded.p', 'wb'))
