#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.1 Transfer Learning
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# You can use textgenrnn for text transfer learnin by training on one text dataset, and then another.

# In[1]:


from textgenrnn import textgenrnn

textgen = textgenrnn()


# In[2]:


file_path = "../datasets/reddit_rarepuppers_politics_2000.txt"

textgen.train_from_file(file_path, new_model=True, num_epochs=10, gen_epochs=10)


# To transfer the learnings, train the `textgenrnn` instance on another dataset without specifying `new_model=True`. You should train the second dataset for fewer epochs.

# In[3]:


file_path = "../datasets/hacker_news_2000.txt"

textgen.train_from_file(file_path, num_epochs=5, gen_epochs=1)


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
