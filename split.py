import argparse
import os
import pandas as pd
import numpy as np


path = os.getcwd() + "/data"
if(not os.path.exists(path)):
    os.mkdir(path)

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--train', action='store', type=float,
                       required=True, help="percentage of training data (i.g. 0.7)")

args = my_parser.parse_args()


df = pd.read_csv('data/parsed.csv')
msk = np.random.rand(len(df)) <= args.train
train = df[msk]
test = df[~msk]
train.to_csv(path+"/train.csv")
test.to_csv(path+"/test.csv")
