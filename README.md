# Abstract
In this project, we are examining article headlines and attempting to quantify how humorous they are. The model uses a continuous scale from 1 to 5 to rank how humorous a particular article headline is. The process and results of this project can be used for applications such as advancing a machine’s understanding of humor in the English language and could help technologies such as voice assistant jokes.

# Problem Statement
The problem involves training a model on thousands of headlines that other people have quantified as humorous or not. In other words, the ground truth is other people’s perceptions of how funny certain headlines are. And we use these values to train the model. 

# Related work
The dataset for this project can be found at https://cs.rochester.edu/u/nhossain/funlines.html \
And the inpsiration for this project is from this paper: https://arxiv.org/pdf/2002.02031.pdf

# Methodology
We used a transformer model and an LSTM model.

# Experiments/evaluation
x

# Results
x

# Examples
x


# Parsing the data
Simply run `python3 parse.py` which generates a csv files with the original headlines as well as the edited ones with the mean grade for humor.
# Split the data
Run `python3 split.py 0.7` to split the data into train and test with %70 of the data for training and %30 for testing in the data/ directory

# FunLines
<input type="text" id="name" name="name"/>
