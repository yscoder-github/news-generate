#!/usr/bin/env python
# coding: utf-8

# # textgenrnn 1.3 Encoding Text
# by [Max Woolf](http://minimaxir.com)
# 
# *Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

# ## Intro
# 
# textgenrnn can also be used to generate sentence vectors much more powerful than traditional word vectors.
# 
# **IMPORTANT NOTE**: The sentence vectors only account for the first `max_length - 1` tokens. (in the pretrained model, that is the first **39 characters**). If you want more robust sentence vectors, train a new model with a very high `max_length` and/or use word-level training.

# In[1]:


from textgenrnn import textgenrnn

textgen = textgenrnn()


# The function `encode_text_vectors` takes the Attention layer output of the model and can use PCA and TSNE to compress it into a more reasonable size.
# 
# The size of the Attention layer is `dim_embeddings + (rnn_size * rnn_layers)`. In the case of the included pretrained model, the size is `100 + (128 * 2) = 356`.
# 
# By default, `encode_text_vectors` uses PCA to project and calibrate this high-dimensional output to the number of provided texts, or 50D, whichever is lower.

# In[2]:


texts = ['Never gonna give you up, never gonna let you down',
            'Never gonna run around and desert you',
            'Never gonna make you cry, never gonna say goodbye',
            'Never gonna tell a lie and hurt you']

word_vector = textgen.encode_text_vectors(texts)

print(word_vector)
print(word_vector.shape)


# Additionally, you can pass `tsne_dims` to further project the texts into 2D or 3D; great for data visualization. (NB: t-SNE is a random-seeded algorithm; for consistent output, set `tsne_seed` to make the output deterministic)

# In[3]:


word_vector = textgen.encode_text_vectors(texts, tsne_dims=2, tsne_seed=123)

print(str(word_vector))
print(word_vector.shape)


# If you want to encode a single text, you'll have to set `pca_dims=None`.

# In[4]:


word_vector = textgen.encode_text_vectors("What is love?", pca_dims=None)

print(str(word_vector)[0:50])
print(word_vector.shape)


# You can also have the model return the `pca` object, which can then be used to learn more about the projection, and/or used in an encoding pipeline to transform any arbitrary text.

# In[5]:


word_vector, pca = textgen.encode_text_vectors(texts, return_pca=True)

print(pca)


# In[6]:


pca.explained_variance_ratio_


# In this case, 56.9% of the variance is explained by the 1st component, and 98.5% of the variance is explained by the first 2 components.

# In[7]:


def transform_text(text, textgen, pca):
    text = textgen.encode_text_vectors(text, pca_dims=None)
    text = pca.transform(text)
    return text

single_encoded_text = transform_text("Never gonna give", textgen, pca)

print(single_encoded_text)


# ## Sentence Vector Similarity

# For example you could calculate pairwise similarity...

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity

word_vectors = textgen.encode_text_vectors(texts)
similarity = cosine_similarity(single_encoded_text, word_vectors)

print(similarity)


# ...or use textgenrnn's native similarity metrics!

# In[9]:


textgen.similarity("Never gonna give", texts)


# By default similarity is calculated using the PCA-transformed values, but you can calculate similarity on the raw values as well if needed.

# In[10]:


textgen.similarity("Never gonna give", texts, use_pca=False)


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
