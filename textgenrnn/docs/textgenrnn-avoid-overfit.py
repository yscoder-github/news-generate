#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.2 Avoiding Overfit
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# You can use textgenrnn for text transfer learnin by training on one text dataset, and then another.


from textgenrnn import textgenrnn

textgen = textgenrnn()




file_path = "../datasets/reddit_rarepuppers_politics_2000.txt"

textgen.reset()
textgen.train_from_file(file_path, new_model=True, num_epochs=5, gen_epochs=5)


# You can specify a `train_size` to train on a subset of sequences, and prevent the neural network from learning *exact* sequences. The remaining data will be used as the validation set.
# 
# Validation loss tests the efficiency of the model on sequences that the model has not seen. Ideally, validation loss should not *increase* after each epoch.


file_path = "../datasets/reddit_rarepuppers_politics_2000.txt"

# textgen.reset()
# textgen.train_from_file(file_path, new_model=True, num_epochs=5, gen_epochs=1, train_size=0.8)



# Additionally, you can add a `dropout`, which drops out that proportion of characters in a sequence for a given epoch. This forces the model to weigh remaining characters more efficiently.
# 
# Only use if `max_length` is not low, and don't set higher than `0.2` or the model may fail to converge!


file_path = "../datasets/reddit_rarepuppers_politics_2000.txt"

textgen.reset()
textgen.train_from_file(file_path, new_model=True, num_epochs=100, gen_epochs=1, train_size=0.8, dropout=0.2)


