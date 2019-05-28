#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.1 Demo
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# textgenrnn is a Python module on top of Keras/TensorFlow which can easily generate text using a pretrained recurrent neural network:

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen = textgenrnn(name="new_model")


fulltext_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/news150w/news2016zh_train.text"
textgen.reset()
textgen.train_from_largetext_file(fulltext_path, new_model=True, num_epochs=1)


# Training on text at the word level works great, although it's strongly recommended to reduce the `max_length` and `max_gen_length` during training!



# textgen.reset()
# textgen.train_from_largetext_file(fulltext_path, new_model=True, num_epochs=1,
#                                   word_level=True,
#                                   max_length=10,
#                                   max_gen_length=50,
#                                   max_words=5000)


