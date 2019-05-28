#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.3.1 Text Cycling Demo
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# In[1]:


from textgenrnn import textgenrnn

textgen = textgenrnn()


# The `generate` function generates `n` text documents at a given temperature:

# In[5]:


textgen.generate(5, temperature=0.2)


# You can pass a `list` of temperatures to `temperatures` instead, and the prediction will cycle through the given temperatures when generating new characters/words. Alternating a low temperature with a high temperature allows the model to access "hidden" knowledge without becoming a complete trainwreck.

# In[3]:


generated_texts = textgen.generate(5, temperature=[0.2, 1.0])


# You can specify a temperature multiple times to repeat it for sequential tokens.

# In[4]:


generated_texts = textgen.generate(5, temperature=[0.2, 0.2, 1.0, 1.0])


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
