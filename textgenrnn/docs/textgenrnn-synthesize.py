#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.5 Model Synthesis
# 
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# You can predict texts from multiple models simultaneously using the `synthesize` function, allowing the creation of texts which incorporate multiple styles without "locking" into a given style.
# 
# You will get better results if the input models are trained with high `dropout` (0.8-0.9)

# In[1]:


from textgenrnn import textgenrnn
from textgenrnn.utils import synthesize, synthesize_to_file

m1 = "gaming"
m2 = "Programmerhumor"


# In[2]:


def create_textgen(model_name):
    return textgenrnn(weights_path='{}_weights.hdf5'.format(model_name),
                     vocab_path='{}_vocab.json'.format(model_name),
                     config_path='{}_config.json'.format(model_name),
                     name=model_name)

model1 = create_textgen(m1)
model2 = create_textgen(m2)


# You can pass a `list` of models to generate from to `synthesize`. The rest of the input parameters are the same as `generate`.

# In[3]:


models_list = [model1, model2]

synthesize(models_list, n=5, progress=False)


# The model generation order is randomized for each creation. It may be worthwhile to double or triple up on models so that the text can generate from the same "model" for multiple tokens.
# 
# e.g. `models_list*3` triples the number of input models, allowing generation strategies such as `[model1, model1, model2, model1, model2, model2]`.

# In[4]:


synthesize(models_list*3, n=5, progress=False)


# You can also `synthesize_to_file`.

# In[5]:


synthesize_to_file(models_list*3, "synthesized.txt", n=10)


# You can also use more than 2 models. One approach is to create a weighted average, for example, create a model that is 1/2 `model1`, 1/4 `model2`, 1/4 `model3`.

# In[6]:


m3 = "PrequelMemes"
model3 = create_textgen(m3)


# In[9]:


models_list2 = [model1, model1, model2, model3]

synthesize(models_list2, n=5, progress=False)


# For character-level models, the models "switch" by default after a list of `stop_tokens`, which by default are a space character or a newline. You can override this behavior by passing `stop_tokens=[]` to a synthesize function, which will cause the model to switch after each character (note: may lead to *creative* results!)
# 
# Word-level models will always switch after each generated token.

# In[10]:


synthesize(models_list2, n=5, progress=False, stop_tokens=[])


# # LICENSE
# 
# MIT License
# 
# Copyright (c) 2019 Max Woolf
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

# In[ ]:




