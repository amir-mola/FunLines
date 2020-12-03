import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import embedding
from utils.cachedir import cache_dir

import numpy as np
import pickle
import random

input_size = 768
output_size = 1

hidden_size = 5
rnn_layers = 3
batch_size = 4
num_epochs = 10
learning_rate = 0.0001


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.gru = torch.nn.GRU(input_size, hidden_size,
                                rnn_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden()
        x, h = self.gru(x, h)
        x = self.fc(F.relu(x[:,-1,:]))
        return x, h

    def init_hidden(self):
        return torch.zeros(rnn_layers, batch_size, hidden_size)


class RNN():
    def __init__(self, train_data, test_data):
        self.model = Net()
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        i = 0
        for _ in range(num_epochs):
            h = self.model.init_hidden()
            for batch in self.train_data:
                x, y = self.get_pair(batch)
                if x.size()[0] != batch_size:
                    continue

                y_hat, h = self.model(x, h)

                loss = self.loss_fn(y_hat, y)

                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()

                if i % (100 / batch_size) == 0:
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
                y_hat, _ = self.model(x)
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
        print('done embedding')
        train = batch_rnn(train, batch_size)
        test = batch_rnn(test, batch_size)
        pickle.dump(train, open(f'{cache_dir}/train_rnn.p', 'wb'))
        pickle.dump(test, open(f'{cache_dir}/test_rnn.p', 'wb'))


def batch_rnn(data, size):
    random.shuffle(data)
    num_batch = len(data)//size

    batches = []
    for i in range(num_batch):
        batch_features = []
        batch_scores = []

        for (states, scores) in data[i*size:(i+1)*size]:
            batch_features.append(states[0])  # Entire embedding
            batch_scores.append(scores)

        batches.append((
            torch.stack(batch_features).squeeze(),
            batch_scores
        ))

    # Putting last batch which is smaller than others
    if len(data[(i+1)*size:]) > 0:
        batch_features = []
        batch_scores = []

        for (states, scores) in data[(i+1)*size:]:
            batch_features.append(states[0])  # Entire embedding
            batch_scores.append(scores)

        batches.append((
            torch.stack(batch_features).squeeze(),
            batch_scores
        ))

    return batches
