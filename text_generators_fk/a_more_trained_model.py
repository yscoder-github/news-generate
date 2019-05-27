#!/usr/bin/env python
# coding: utf-8

# ## Importing Dependencies

# In[53]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils


# ## Loading of Data

# In[54]:


text = (open("./sonnets.txt").read())
text=text.lower()


# ## Creating character/word mappings

# In[55]:


characters = sorted(list(set(text)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}


# ## Data pre-processing

# In[56]:


X = []
Y = []
length = len(text)
seq_length = 100

for i in range(0, length - seq_length, 1):
    sequence = text[i : i + seq_length]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])


# In[57]:


X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)


# ## A more trained model

# In[58]:


model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


model.fit(X_modified, Y_modified, epochs=100, batch_size=50)

model.save_weights('text_generator_400_0.2_400_0.2_100.h5')


# In[59]:


model.load_weights('./models/text_generator_400_0.2_400_0.2_100.h5')


# ## Generating Text

# In[60]:


string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]


# In[63]:


#combining text
txt=""
for char in full_string:
    txt = txt+char
txt


# In[ ]:




