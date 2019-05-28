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


# ## Generate Text
# 
# The `generate` function generates `n` text documents:


textgen.generate(5)


# In addition, you can set the `temperature` to modify the amount of creativity (default 0.5; I do not recommend setting to more than 1.0), set a `prefix` to force the document to start with certain characters and generate characters accordingly, and set a `return_as_list` flag (default False) to use the generated texts elsewhere in your application (e.g. as an API)

generated_texts = textgen.generate(n=5, prefix="Trump", temperature=0.2, return_as_list=True)
print(generated_texts)


# Using `generate_samples()` is a great way to test the model at different temperatures.


textgen.generate_samples()


# You may also `generate_to_file()` to make the generated texts easier to copy/paste to other sources (e.g. blog/social media):



textgen.generate_to_file('textgenrnn_texts.txt', n=5)


# ## Train on New Text
# 
# As shown above, the results on the pretrained model vary greatly, since it's a lot of data compressed in a small form. The `train_on_texts` function fine-tunes the model on a new dataset.


texts = ['Never gonna give you up, never gonna let you down',
            'Never gonna run around and desert you',
            'Never gonna make you cry, never gonna say goodbye',
            'Never gonna tell a lie and hurt you']

textgen.train_on_texts(texts, num_epochs=2,  gen_epochs=2)


# Although the network was only trained on 4 texts, the original network still transfers the latent knowledge of all modern grammar and can incorporate that knowledge into generated texts, which becomes evident at higher temperatures or when using a prefix containign a character not present in the original dataset.

# You can reset a trained model back to the original state by calling `reset()`.



textgen.reset()


# Included in the repository is a `hacker-news-2000.txt` file containing a list of the Top 2000 [Hacker News](https://news.ycombinator.com/news) submissions by score. Let's retrain the model using that dataset.
# 
# For this example, I only will use a single epoch to demonstrate how easily the model learns with just one pass of the data: I recommend leaving the default of 50 epochs, or set it even higher for complex datasets. On my 2016 15" MacBook Pro (quad-core Skylake CPU), the dataset trains at about 1.5 minutes per epoch.
textgen.train_from_file('../datasets/hacker_news_2000.txt', num_epochs=1)


# Now, we can create very distinctly-HN titles, even with the very little amount of training, thanks to the pre-trained nature of the textgenrnn:



textgen.generate(5, prefix="Apple")


# Other runtime parameters for `train_on_text` and `train_from_file` are:
# 
# * `num_epochs`: Number of epochs to train for (default: 50)
# * `gen_epochs`: Number of epochs to run between generating sample outputs; good for measuring model progress (default: 1)
# * `batch_size`: Batch size for training; may want to increase if running on a GPU for faster training (default: 128)
# * `train_size`: Random proportion of sequence samples to keep: good for controlling overfitting. The rest will be used to train as the validation set. (default: 1.0/all). To disable training on the validation set (for speed), set `validation=False`.
# * `dropout`: Random number of tokens to ignore each epoch. Good for controlling overfitting/making more resilient against typos, but setting too high will cause network to converge prematurely. (default: 0.0)
# * `is_csv`: Use with `train_from_file` if the source file is a one-column CSV (e.g. an export from BigQuery or Google Sheets) for proper quote/newline escaping.

# ## Save and Load the Model
# 
# The model saves the weights automatically after each epoch, or you can call `save()` and give a HDF5 filename. Those weights can then be loaded into a new textgenrnn model by specifying a path to the weights on construction. (Or use `load()` for an existing textgenrnn object).


textgen_2 = textgenrnn('textgenrnn_weights.hdf5')
textgen_2.generate_samples()



textgen.model.get_layer('rnn_1').get_weights()[0] == textgen_2.model.get_layer('rnn_1').get_weights()[0]


# Indeed, the weights between the original model and the new model are equivalent.
# 
# You can use this functionality to load models from others which have been trained on larger datasets with many more epochs (and the model weights are small enough to fit in an email!).



textgen = textgenrnn('../weights/hacker_news.hdf5')
textgen.generate_samples(temperatures=[0.2, 0.5, 1.0, 1.2, 1.5])


# ## Training a New Model
# 
# You can train a new model using any modern RNN architecture you want by calling `train_new_model` if supplying texts, or adding a `new_model=True` parameter if training from a file. If you do, the model will save a `config` file and a `vocab` file in addition to the weights, and those must be also loaded into a `textgenrnn` instances.
# 
# The config parameters available are:
# 
# * `word_level`: Whether to train the model at the word level (default: False)
# * `rnn_layers`: Number of recurrent LSTM layers in the model (default: 2)
# * `rnn_size`: Number of cells in each LSTM layer (default: 128)
# * `rnn_bidirectional`: Whether to use Bidirectional LSTMs, which account for sequences both forwards and backwards. Recommended if the input text follows a specific schema. (default: False)
# * `max_length`: Maximum number of previous characters/words to use before predicting the next token. This value should be reduced for word-level models (default: 40)
# * `max_words`: Maximum number of words (by frequency) to consider for training (default: 10000)
# * `dim_embeddings`: Dimensionality of the character/word embeddings (default: 100)
# 
# You can also specify a `name` when creating a textgenrnn instance which will help name the output weights/config/vocab appropriately.



textgen = textgenrnn(name="new_model")




textgen.reset()
textgen.train_from_file('../datasets/hacker_news_2000.txt',
                        new_model=True,
                        rnn_bidirectional=True,
                        rnn_size=64,
                        dim_embeddings=300,
                        num_epochs=1)

print(textgen.model.summary())



textgen_2 = textgenrnn(weights_path='new_model_weights.hdf5',
                       vocab_path='new_model_vocab.json',
                       config_path='new_model_config.json')

textgen_2.generate_samples()


# ## Train on Single Large Text
# 
# Although textgenrnn is intended to be trained on text documents, you can train it on a large text block using `train_from_largetext_file` (which loads the entire file and processes it as if it were a single document) and it should work fine. This is akin to more traditional char-rnn tutorials.
# 
# Training a new model is recommended (and is the default). When calling `generate`, you may want to increase the value of `max_gen_length`.



from keras.utils.data_utils import get_file

fulltext_path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

textgen.reset()
textgen.train_from_largetext_file(fulltext_path, new_model=True, num_epochs=1)


# Training on text at the word level works great, although it's strongly recommended to reduce the `max_length` and `max_gen_length` during training!



textgen.reset()
textgen.train_from_largetext_file(fulltext_path, new_model=True, num_epochs=1,
                                  word_level=True,
                                  max_length=10,
                                  max_gen_length=50,
                                  max_words=5000)


