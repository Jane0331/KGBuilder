#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import re
import copy
import math
import chardet
import optparse
import numpy as np


def load_stopwords():
    global stopwords
    with open('stopwords.txt') as f:
        for line in f.readlines():
            line=line.rstrip()
            stopwords.add(line)
    print 'Load %d stopwords' % len(stopwords)


def load_targetwords(target_list):
    global targetwords
    for targetfile in target_list:
        with open(targetfile) as fin:
            for line in fin.readlines():
                line=line.rstrip()
                if line:
                    for word in line.split(' '):
                        if opts.lower:
                            targetwords.add(word.lower())
                        else:
                            targetwords.add(word)
    print 'Load '+str(len(targetwords))+' words from '+str(target_list)


def isLegal(w):
    if len(w)<=1:
        return False
    if targetwords and not w in targetwords:
        return False
    if w.isdigit():
        return False
    if w.upper()!=w and (w in stopwords or w.lower() in stopwords):
        return False
    for a in w:
        if chardet.detect(a)['encoding']!='ascii':
            return False
    return True


def precess_words(line,word_dic,word_locdic,max_fre):
    words=[]
    twords=re.split(r'([\s~`!@#\$%\^&\*\(\)\-_\+=\{\}\[\]\|\\:;\"\'<>,\?/])', line)
    for w in twords:
        for tw in w.split():
            words.append(tw)

    for i in range(len(words)):
        w=words[i]
        if isLegal(w):
            if opts.lower:
                w=w.lower()
            if word_dic.has_key(w):
                word_dic[w]+=1
            else:
                word_dic[w]=1
            if word_dic[w]>max_fre:
                max_fre=word_dic[w]

            if not word_locdic.has_key(w):
                word_locdic[w]=[0,0,0,0]
            if len(words)>1:
                if i==0:
                    word_locdic[w][0]+=1
                elif i==len(words)-1:
                    word_locdic[w][2]+=1
                else:
                    word_locdic[w][1]+=1
            else:
                word_locdic[w][3]+=1
    return word_dic,word_locdic,max_fre


def process_dic(dicfile):
    KBE_dic={}
    word_dic={}
    word_locdic={}
    max_fre=0
    with open(dicfile, 'r') as fin:
        for line in fin.readlines():
            line=line.rstrip()
            word_dic,word_locdic,max_fre=precess_words(line,word_dic,word_locdic,max_fre)
    print '%s: maximum appearing times is %d' % (dicfile, max_fre)

    temp=math.log(max_fre,10)/(opts.beta-opts.alpha)
    for word in word_dic:
        times=word_dic[word]
        fKBE=math.log(times,10)/temp+opts.alpha
        lKBE=[fKBE*t/times for t in word_locdic[word]]
        KBE_dic[word]=[opts.beta, lKBE[0], lKBE[1], lKBE[2], lKBE[3]] #nKBE+lKBE
    return KBE_dic


def add_random(embeddings):
    for word in embeddings:
        for j in range(opts.origin_dim):
            embeddings[word].append(np.random.normal(0,0.4))
    return embeddings


def add_pre(embeddings,zeros):
    with open(opts.pre_emb) as fin:
        for line in fin.readlines():
            line=line.rstrip()
            pline=line.split(' ')
            word=pline[0]
            if targetwords and not w in targetwords:
                continue
            if not len(pline)==opts.origin_dim+1:
                raise Exception('Word embedding dimension error for \"%s\"' % word)
            if not word in embeddings:
                embeddings[word]=copy.deepcopy(zeros)
            for e in pline[1:]:
                embeddings[word].append(float(e))


def merge(KBE_dics):
    embeddings={}
    zeros=[]
    dimen=0
    for KBE_dic in KBE_dics:
        tdimen=0
        for word in KBE_dic:
            tdimen=len(KBE_dic[word])
            if not embeddings.has_key(word):
                embeddings[word]=copy.deepcopy(zeros)
            elif len(embeddings[word])>dimen:
                print '[Dup] '+word
                continue
            for j in KBE_dic[word]:
                embeddings[word].append(j)
        
        dimen+=tdimen

        for word in embeddings:
            if len(embeddings[word])!=dimen:
                for j in range(tdimen):
                    embeddings[word].append(0)
        for j in range(tdimen):
            zeros.append(0)
    print '%d target words can be found in dictionaries' % len(embeddings)

    #words not appearing any dictionary
    for word in targetwords:
        if not word in embeddings:
            embeddings[word]=copy.deepcopy(zeros)

    if not opts.pre_emb:
        embeddings=add_random(embeddings)
    else:
        embeddings=add_pre(embeddings,zeros)
    return embeddings


def write_dic(embeddings):
    embeddings_list=embeddings.items()
    embeddings_list.sort()
    with open(opts.emb, 'w') as f:
        for e in embeddings_list:
            f.write(e[0])
            for d in e[1]:
                f.write(' '+str(d))
            f.write('\n')


optparser = optparse.OptionParser()
optparser.add_option(
    "-c", "--dicpath", default="",
    help="Lacation of all dictionaries"
)
optparser.add_option(
    "-e", "--emb", default="",
    help="Output embedding file location"
)
optparser.add_option(
    "-a", "--alpha", default="1.00",
    type='float', help="The value of fKBE when t_i=1"
)
optparser.add_option(
    "-b", "--beta", default="2.00",
    type='float', help="The maximum value of KBE"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-o", "--origin_dim", default="0",
    type='int', help="Original word embedding dimension (this must equal to the embedding dimension in pre_emb if it is given, otherwise, this is the dimension of original embeddings which initialize randomly)"
)
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert os.path.isdir(opts.dicpath)
assert not opts.pre_emb or os.path.isfile(opts.pre_emb)
assert not opts.pre_emb or opts.origin_dim > 0
assert not opts.train or os.path.isfile(opts.train)
assert not opts.dev or os.path.isfile(opts.dev)
assert not opts.test or os.path.isfile(opts.test)
if not opts.pre_emb and opts.origin_dim==0:
    opts.origin_dim = 50
opts.lower=(opts.lower==1)
#print opts

stopwords=set()
targetwords=set()
KBE_dics=[]

load_stopwords()

target_list=[file for file in filter(lambda x:x, [opts.train, opts.dev, opts.test])]
load_targetwords(target_list)

file_list=[os.path.join(opts.dicpath,file) for file in filter(lambda x:x.endswith('.dic'), os.listdir(opts.dicpath))]
for dicfile in file_list:
    KBE_dic=process_dic(dicfile)
    KBE_dics.append(KBE_dic)

embeddings=merge(KBE_dics)
write_dic(embeddings)
