"""
This script will train a RNN on a dataset consisting of comments on articles for The New York Times using TensorFlow.
link to data: https://www.kaggle.com/datasets/aashita/nyt-comments
This model will be used to peform text generation. 

please make sure to install the necessary packages by running setup script like so:
bash setup.sh

"""

# Import packages
# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# for saving tokenizer
from joblib import dump

# Defining a function which cleans the text. Removes punctuation and changes to lower case.
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

# Defining a function which produces a list of input sequences.
# every sentence gets represented by a sequence of numbers where each number corresponds to a specific token.
# In this way sentence structure is preserved.
def get_sequence_of_tokens(tokenizer, corpus):
    # convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

# Defining a function which pads the sequences since neural networks require inputs which are of the same lenght.
# nts: addad total_words to arguments
def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest one
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

# Defining the structure of the RNN model
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    # Adding dropout layer to avoid overfitting
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model


def load_data():
    data_dir = os.path.join("data")
    # loading the data one at a time and appending only the comments to list of data.
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            article_df = pd.read_csv(data_dir + "/" + filename)
            all_comments.extend(list(article_df["commentBody"].values))
    corpus = [clean_text(x) for x in all_comments]

    # !!!
    # Pulling a small random sample of comments to train the model on as a proof of concept.
    # Feel free to alter the sample size or ignore the following two lines of code by adding "#" in front. 
    import random
    corpus = random.sample(all_comments, 100)
    #!!!

    return corpus

def tokenize(corpus):
    tokenizer = Tokenizer()
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    inp_sequences = get_sequence_of_tokens(tokenizer, corpus)
    # NTS: inp_sequences, total_words = get_sequence_of_tokens(tokenizer, corpus)
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    # saving max_sequence_len as txt file because it is needed for the text generation script.
    txtfile = open("max_sequence_len.txt", "w")
    txtfile.write(str(max_sequence_len))
    txtfile.close()

    # saving tokenizer in "models" folder
    tokenizer_path = os.path.join("models", "tokenizer.joblib")
    dump(tokenizer, tokenizer_path)

    return total_words, max_sequence_len, predictors, label

def fit_model(max_sequence_len, predictors, label, total_words):
    # model = create_model(max_sequence_len, predictors, label, total_words)
    model = create_model(max_sequence_len, total_words)
    model.summary()
    history = model.fit(predictors, 
                    label, 
                    epochs=100,
                    batch_size=128, 
                    verbose=1)
    # saving the RNN model
    model_path = os.path.join("models", "RNN-model.keras")
    model.save(model_path)

def main():
    corpus = load_data()
    total_words, max_sequence_len, predictors, label = tokenize(corpus)
    fit_model(max_sequence_len, predictors, label, total_words)


if __name__=="__main__":
    main()
