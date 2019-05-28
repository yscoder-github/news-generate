#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.1 Training with Context
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# textgenrnn is a Python module on top of Keras/TensorFlow which can easily generate text using a pretrained recurrent neural network:

# In[1]:


from textgenrnn import textgenrnn

textgen = textgenrnn()


# ## Train on New Text
# 
# Here's textgenrnn trained on a dataset of the Top 1000 submissions to /r/politics and /r/rarepuppers during 2017: two subreddits with drastically different styles! How well can a new model learn these styles?

# In[2]:


file_path = "../datasets/reddit_rarepuppers_politics_2000.txt"

textgen.reset()
textgen.train_from_file(file_path, new_model=True, num_epochs=10, gen_epochs=5)


# And here's the same dataset, trained with the context labels indicating the source subreddit. If using `train_from_file`, the input file must be a 2-column CSV, with the first column containing the texts and the second containing the labels.
# 
# The `output_loss` is the loss along the text-only path. In this example, the context-label model has a slightly lower loss than the text-only training. (the more disparate the texts, the more helpful providing context will help)

# In[2]:


file_path = "../datasets/reddit_rarepuppers_politics_2000_context.csv"

textgen.reset()
textgen.train_from_file(file_path, new_model=True, num_epochs=10, gen_epochs=5, context=True)


# You can use `train_on_texts` with context labels as well by providing a list of context labels of the same size as the texts.

# In[ ]:


texts = ['Never gonna give you up, never gonna let you down',
            'Never gonna run around and desert you',
            'Never gonna make you cry, never gonna say goodbye',
            'Never gonna tell a lie and hurt you']

context_labels = ['Verse 1', 'Verse 1', 'Verse 2', 'Verse 2']

textgen.reset()
textgen.train_new_model(texts, context_labels=context_labels, max_length=5, gen_epochs=5, num_epochs=10)


# # LICENSE
# 
# MIT License
# 
# Copyright (c) 2018 Max Woolf
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
