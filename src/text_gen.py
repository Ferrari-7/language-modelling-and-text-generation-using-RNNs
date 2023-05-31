"""
The following script does the following:
- loads the text generation model trained in the trainRNN.py script.
- Generates text from a user-suggested prompt. 

Instructions: 
This script uses argparse and takes two arguments. 
When running the script, please state the prompt word/words from which the text will be generated.
Optionally, the user can provide the length of the text string following the prompt. The code will default to twenty if no argument is provided.

example:

python src/text_gen.py --prompt democracy --lenght 15

"""
# Import packages
from joblib import load
import argparse
import os
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Defining user arguments
def input_parse():  
    # initialize the parser 
    parser = argparse.ArgumentParser() 
    # add arguments 
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Please enter your prompt")
    parser.add_argument("-l", "--lenght", type=int, default="20", help="Enter desired number of words following prompt")
    # parse the arguments from command line
    args = vars(parser.parse_args())
    prompt = args["prompt"]
    length = args["lenght"]
    return prompt, length

# Loading the RNN model created in the trainRNN.py script.
def load_model():
    model_path = os.path.join("models", "RNN-model.keras")
    model = tf.keras.models.load_model(model_path)
    return model


def generate_text(prompt, length, model):
    # loading the saved tokenizer model
    tokenizer_path = os.path.join("models", "tokenizer.joblib")
    tokenizer = load(tokenizer_path)

    # loading max_sequence_len which has been stored in txt file.
    txtfile = open("max_sequence_len.txt", "r")
    max_sequence_len = int(txtfile.read())
    txtfile.close()

    for _ in range(length):
        token_list = tokenizer.texts_to_sequences([prompt])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        prompt += " "+output_word
    return prompt.title()

def save_gen_text(prompt, length, model):
    generated_text = generate_text(prompt, length, model)
    print(generated_text)

    txt_path = os.path.join("examples", prompt + "_" + str(length) + ".txt")
    txt_file = open(txt_path, "w")
    txt_file.write(generated_text)
    txt_file.close()

def main():
    prompt, length = input_parse()
    model = load_model()
    save_gen_text(prompt, length, model)
    #print(generate_text(prompt, length, model))

if __name__=="__main__":
    main()
