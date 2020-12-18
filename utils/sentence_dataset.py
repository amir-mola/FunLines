import torch
from embedding import create_dataset

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(SentenceDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return list(self.data.loc[idx])



def main():
    train, test = create_dataset('data/train_lines.csv')
    trainset = SentenceDataset(train)
    x = trainset.__len__()
    print(x)
    print(trainset[0])


main()