import pandas as pd
import pickle
import matplotlib.pyplot as plt
from models import LinearModel, RNN, TransformerModel, TransformerwWordModel
from utils.cachedir import cache_dir


def main():
    # model = TransformerModel()
    model = TransformerwWordModel()
    model.train()
    plt.plot([i for i in range(len(model.train_losses))], model.train_losses, label="train")
    plt.plot([i for i in range(len(model.val_losses))], model.val_losses, label="val")
    plt.legend()
    plt.title("RMSE VS. Epoch")
    plt.show()


if __name__ == '__main__':
    main()
