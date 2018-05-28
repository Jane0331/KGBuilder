#!/bin/sh
input=$1
output=$2
model_ner=$3
model_re=$4

output_ner=${input}.entity

#named entity recognition
python ./NER/tagger.py --model $model_ner --input $input --output $output_ner
#relation extraction


