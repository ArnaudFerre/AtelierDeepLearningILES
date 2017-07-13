#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre@limsi.fr
Description: tools functions for Deep Learning program with Keras lib.
License: DSSL
"""

#######################################################################################################
# Importations
#######################################################################################################
import numpy
from scipy import spatial
#import os, sys, time, json, pickle

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



def getNearestWords(w, d_w2vData, threshold = 0.9):

    d_distances = dict()
    for word in d_w2vData.keys():
        d_distances[word] = getCosSimilarity(d_w2vData[w], d_w2vData[word])

    print(w)
    for word in d_distances.keys():
        if d_distances[word] > threshold:
            print("     " + word + " --> " + str(d_distances[word]))



def loadTestOrTrains(pathTest):
    l_test = list()
    file = open(pathTest)
    i=0
    for line in file:
        line = line.split()
        if len(line) > 1:
            l_test.append(list())
            l_test[i].append(line[0].lower())
            l_test[i].append(line[1])
            i+=1
    return l_test
    

def getXY_data(lc_trainOrTest, d_w2vData):
    X = list()
    Y = list()
    print("Tokens sans vecteur : ")
    for data in lc_trainOrTest:
        try:
            wordVec = d_w2vData[data[0].lower()]

            if data[1] == "Disease":  # = (0,1)
                labelVec = numpy.array([0, 1])
            elif data[1] == "O":  # = (1,0)
                labelVec = numpy.array([1, 0])

            X.append(wordVec)
            Y.append(labelVec)

        except:
            print(data[0], " ", end='')

    # Convertir liste de vecteurs en numpyArray :
    X = numpy.array(X)
    print("X.shape = ", X.shape)
    Y = numpy.array(Y)
    print("Y.shape = ", Y.shape)

    return X, Y


def norm(classes):
    i = 0
    for couple in classes:
        if couple[0] == max(couple[0], couple[1]):
            classes[i] = numpy.array([1,0])
        else:
            classes[i] = numpy.array([0,1])
        i+=1
    return classes


def eval(NormClasses, Y_test):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(NormClasses)):
        if numpy.array_equal(NormClasses[i], numpy.array([0,1])): #Disease identifiée
            if numpy.array_equal(NormClasses[i], Y_test[i]): #Si correct
                tp += 1
            else: #Si incorrect
                fp += 1
        elif numpy.array_equal(NormClasses[i], numpy.array([1,0])): #O identifiée
            if numpy.array_equal(NormClasses[i], Y_test[i]):
                tn += 1
            else:
                fn += 1
        else:
            print("anormal...")
    return tp,fp,tn,fn, (tp+fp+tn+fn)


#######################################################################################################
#######################################################################################################
if __name__ == '__main__':

    ###################################################
    # Chargement des données :
    ###################################################

    # Chargement des embeddings :
    pathFile = "word2vecData_embeddings_dim100.txt"
    d_w2vData = loadEmbeddings(pathFile)

    # Chargement des données textuelles :
    pathTestFile = "test.txt"
    pathTrainFile = "train.txt"
    lc_test = loadTestOrTrains(pathTestFile)
    lc_train = loadTestOrTrains(pathTrainFile)

    # Mise en forme des données pour entraînement :
    X_train, Y_train = getXY_data(lc_train, d_w2vData)
    X_test, Y_test = getXY_data(lc_test, d_w2vData)

    print("\n\n")

    ###################################################
    # Keras :
    ###################################################

    # Initialiser un réseau de neurones :
    from keras.models import Sequential
    network = Sequential()

    # Nb couches et neurones
    from keras.layers import Dense, Activation
    network.add(Dense(units=100, input_dim=100))
    network.add(Activation('relu')) # 'relu' mieux que sigmoïd en général
    """
    # On peut rajouter des couches facilement :
    network.add(Dense(units=100))
    network.add(Activation('relu'))
    """
    network.add(Dense(units=2))
    network.add(Activation('softmax'))

    #Compilation du réseau :
    network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #entraînement
    network.fit(X_train, Y_train)

    #Prediction
    classes = network.predict(X_test, batch_size=128)

    print("\n\n")

    ###################################################
    # Normalisation des prédictions (pour n'avoir que [0,1] ou [1,0])
    ###################################################
    result = norm(classes)

    ###################################################
    # Evaluation :
    ###################################################
    tp, fp, tn, fn, total = eval(classes, Y_test)

    print("tp:",tp," fp:",fp," tn:",tn," fn:",fn, " total:",(tp+fp+tn+fn))
    precision = float(tp) / (tp + fp)
    print("precision : ", precision)
    rappel = float(tp) / (tp + fn)
    print("rappel : ", rappel)
    Fmesure = 2*precision*rappel / (precision + rappel)
    print("F-mesure : ", Fmesure)


    ###################################################
    # Bonus :
    ###################################################
    # Si vous voulez tester le représentation vectorielle de Word2Vec,
    # vous pouvez utilisez les fonctions suivantes :

    """
    getCosSimilarityBetweenWords("substance", "dna", d_w2vData)

    getNearestWords("dna", d_w2vData, 0.95)
    """


