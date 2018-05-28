### KGBuilder
KGBuilder is an implementation of a scientific domain knowledge graph builder. It combines domain knowledge with neural networks to overcome the difficulities of in-domain knowledge graph construction, and finally obtains state-of-the-art performance in its subtasks, namely named entity recognition and relation extraction. Details about the tool can be found at: http://

## Initial setup
To use the tool, you need Python 2.7, with Numpy, Theano and TensorFlow installed.

## Train models
To train your own model, you need to do 3 steps.
#Generate embeddings
[STEP1] Use the ./NER/dic2embed.py script and provide the location of all dictionaries, and the training, development and testing set.
```
./dic2embed.py --dicpath ./ --train train.txt --dev dev.txt --test test.txt --output embed.txt
```
This script generates word embeddings through our KBE method. All of your dictionaries should be put in the same folder with the extension ".dic", and follow the format: each mention is on a separate line. Just put mentions having more than one words, like "bacillus coli", on one line.
The training, development and testing set have to follow the format: each word is on a separate line, and there is an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the second one being the named entity. Tags have to be given in the IOB format.

[STEP2] Use the ./NER/train.py script and provide the location of the training, development and testing set:
```
./train.py --train train.txt --dev dev.txt --test test.txt --pre_emb embed.txt
```
The training script will automatically give a name to the model and store it in ./NER/models/.

[STEP3]

There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run the above three scripts with "--help".

## Build Graph
The fastest way to use the tagger is to use one of the pretrained models:
```
./tagger.py --model models/english/ --input input.txt --output output.txt
```
The input file should contain one sentence by line, and they have to be tokenized. Otherwise, the tagger will perform poorly.
