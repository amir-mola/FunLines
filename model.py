import torch
from torch.autograd import Variable
import torch.nn.functional as F

import embedding
import numpy as np
import pickle
import os

input_size = 768 * 43
output_size = 1

hidden_size = 5
batch_size = 4
num_epochs = 500
learning_rate = 0.00001

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



def get_pair(batch):
    (x1, x2, y) = batch
    x2 = torch.unsqueeze(x2, 1)
    x = torch.cat((x1, x2), 1)
    x = torch.flatten(x, 1)
    y = torch.Tensor(y)
    y = torch.unsqueeze(y, 1)
    return Variable(x), Variable(y)


def train(model, loss_fn, learning_rate, train_data):
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    i = 0
    for _ in range(num_epochs):
        for batch in train_data:
            x, y = get_pair(batch)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 1000 == 0:
                print(loss.item())

            i += 1


def test(model, loss_fn, test_data):
    model.eval()

    tolerance = 0.2
    correct = 0
    total = 0
    losses = []

    # In case of python < 3.5
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    with torch.no_grad():
        for batch in test_data:
            x, y = get_pair(batch)
            y_hat = model(x)
            losses.append(loss_fn(y_hat, y))

            for b in range(batch_size):
                yb = y[b].item()
                yhb = y_hat[b].item()
                if isclose(yb, yhb, rel_tol=tolerance):
                    correct += 1
                total += 1

    print(f'Test accuracy: {correct / total}')
    print(f'Test loss: {np.mean(losses)}')


def load_data():
    train, test = embedding.create_dataset('train_funlines.csv')
    print('done splitting')
    train = embedding.embed(train)
    test = embedding.embed(test)
    train = embedding.create_batch(train, batch_size)
    test = embedding.create_batch(test, batch_size)
    print('done embedding')
    pickle.dump(train, open('data/train.p', 'wb'))
    pickle.dump(test, open('data/test.p', 'wb'))


def main():
    # load_data() # comment this out to re-use last split
    train_data = pickle.load(open('data/train.p', 'rb'))
    test_data = pickle.load(open('data/test.p', 'rb'))
    print('done loading')

    model = Net()
    loss_fn = torch.nn.MSELoss()
    train(model, loss_fn, learning_rate, train_data)
    test(model, loss_fn, test_data)


if __name__ == '__main__':
    main()
