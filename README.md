[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10587783&assignment_repo_type=AssignmentRepo)
# Language modelling and text generation using RNNs

## Description

This repository contains two scripts. 
- One which trains a RNN model on a dataset containing comments from *The New York Times*. Link to data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).  
- The second peforms text generation using this model based on a user-suggested prompt. It then saves the generated text as a txt file to the folder **examples**

**NB** This script is a proof of concept and will only be trained on a random sample of a 100 comments. For better results, please alter the sample size or use # to ignore the lines of the script which takes a random sample. It is marked with "!!!".

| | description |
| --- | --- | 
| data | empty folder where the user can store the data that the model will be trained on |
| examples | folder containing txt files of generated text | 
| models | folder containing the trained RNN model as well as a tokenizer. |
| src | folder containing the two .py scripts. | 
| max_sequence_len.txt | txt file produced by the script which trains the RNN model. Will be used in the text generation script |
| requirements.txt | contains list of necessary packages for running the scripts |
| run_train.sh | shell file which trains and saves a RNN model on the contents of data folder. |
| run_text_gen.sh | shell file which peforms text generation based on the trained models and a user-suggested prompt. |
| setup.sh | shell file which opens a virtual enviroment and installs necessary packages listed in requirements.txt |

## User instructions
1. add csv files to folder called data. Link to data [here](https://www.kaggle.com/datasets/aashita/nyt-comments). The code in this repository has been trained on a random sampling of 100 comments. I chose only to include one csv file (“CommentsApril2017.csv”).

2. Use **setup.sh** to install the necessary packages required to run the scripts like so: 

`bash setup.sh`

3. Use **run_train.sh** to train a RNN network and save the models. The user may increase the number of comments used in the sample or remove this restraint all together. This will produce a better functioning text generator but will require more time and computational resources. The section of code which pulls a random sample is marked with “!!!”

`bash run_train.sh`

4. Use **run_txt_gen.sh** to peform text generation. Please note that the model in this repository has only been trained on a random sample of 100 comments. This script uses argparse. Please inform the prompt from which you wish to generate the text. Additionally, add the lenght of the generated text. The example below demonstrates the format: 

`bash run_text_gen.sh america 15`

## Discussion

RNN’s (Recurrent Neural Networks) are especially suited for neural language models. This is due to the fact that RNN’s address the fact that language is sequential and temporarily ordered. At each step in the training process the RNN takes both the given input and the preceding outputs into account. To avoid the vanishing gradient problem, where increased distance between input and previous outputs results in a diminished ability to learn, the model in this repository has a LSTM layer (Long short term memory). This makes the RNN able to learn long-term dependencies.

