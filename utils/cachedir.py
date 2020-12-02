import os

cache_dir = 'cache'
cache_path = f'{os.getcwd()}/{cache_dir}'
if(not os.path.exists(cache_path)):
    os.mkdir(cache_path)
