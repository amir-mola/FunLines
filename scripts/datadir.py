import os

data_dir = 'data'
data_path = f'{os.getcwd()}/{data_dir}'
if(not os.path.exists(data_path)):
    os.mkdir(data_path)