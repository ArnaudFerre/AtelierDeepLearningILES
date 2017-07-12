#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre@limsi.fr
Description: tools functions
License: DSSL
"""


# import modules & set up logging
import logging, os, sys, time, re, pickle, json, numpy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from scipy import spatial



#######################################################################################################
# Functions
#######################################################################################################
def loadEmbeddings(pathFile):
    d_w2vData = dict()

    file = open(pathFile, "r")

    for line in file:
        l_data = line.split()
        d_w2vData[l_data[0]] = numpy.zeros(len(l_data)-1)

        i=0
        for val in l_data:
            if i>0:
                d_w2vData[l_data[0]][i-1] = float(val)
            i+=1

    return d_w2vData



def getCosSimilarity(vec1, vec2):
    result = 0
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html
    try:
        result = 1 - spatial.distance.cosine(vec1, vec2)
    except:
        print("ERROR during cosinus similarity calculation...")
        result = None
    return result
def getCosSimilarityBetweenWords(w1, w2, d_w2vData):
    result = getCosSimilarity(d_w2vData[w1], d_w2vData[w2])
    print(w1, "//", w2, result)
    return result



def getNearearWords(w, d_w2vData, threshold = 0.9):

    d_distances = dict()
    for word in d_w2vData.keys():
        d_distances[word] = getCosSimilarity(d_w2vData[w], d_w2vData[word])

    print(w)
    for word in d_distances.keys():
        if d_distances[word] > threshold:
            print("     " + word + " --> " + str(d_distances[word]))





#######################################################################################################
#######################################################################################################
if __name__ == '__main__':

    pathFile = "word2vecData_embeddings_dim100.txt"
    d_w2vData = loadEmbeddings(pathFile)

    for word in d_w2vData.keys():
        print(word, len(d_w2vData[word]))

    #Chargement Train
    X_train = list()
    Y_train = list()
    file = open("train.txt", "r")
    for line in file:
        line = line.split()

        try:
            word = line[0].lower()
            label = line[1].rstrip('\n')

            wordVec = d_w2vData[word]

            if label == "Disease":
                labelVec = numpy.zeros(2)
                labelVec[1] = 1
            else:
                labelVec = numpy.zeros(2)
                labelVec[0] = 1

            X_train.append(wordVec)
            Y_train.append(labelVec)

        except:
            print("mot sans vecteur..." + word),

    print(len(X_train))
    print(len(Y_train))



    #Convertir liste de vecteurs en numpyArray :
    X_train = numpy.array(X_train)
    print(X_train, X_train.shape)

    Y_train = numpy.array(Y_train)
    print(Y_train, Y_train.shape)





    """
    getCosSimilarityBetweenWords("substance", "dna", d_w2vData)

    getNearearWords("dna", d_w2vData, 0.9)
    """



    #
    from keras.models import Sequential
    network = Sequential()

    # Nb couches et neurones
    from keras.layers import Dense, Activation
    network.add(Dense(units=100, input_dim=100))
    network.add(Activation('relu')) # Pourquoi des mieux que d'autres ??? relu mieux que sigmoïd en général !
    network.add(Dense(units=2))
    network.add(Activation('softmax'))

    #Compilation du réseau :
    network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #entraînement
    network.fit(X_train, Y_train)



    #Prediction
    classes = network.


