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
        d_w2vData[l_data[0]] = numpy.zeros(len(l_data))

        i=0
        for val in l_data:
            if i>0:
                d_w2vData[l_data[0]][i] = float(val)
            i+=1

    return d_w2vData



def getCosSimilarity(vec1, vec2):
    result = 0
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html
    try:
        result = 1 - spatial.distance.cosine(vec1, vec2)
    except:
        print "ERROR during cosinus similarity calculation..."
        result = None
    return result
def getCosSimilarityBetweenWords(w1, w2, d_w2vData):
    result = getCosSimilarity(d_w2vData[w1], d_w2vData[w2])
    print w1, "//", w2, result
    return result



def getNearearWords(w, d_w2vData, threshold = 0.9):

    d_distances = dict()
    for word in d_w2vData.keys():
        d_distances[word] = getCosSimilarity(d_w2vData[w], d_w2vData[word])

    print w
    for word in d_distances.keys():
        if d_distances[word] > threshold:
            print("     " + word + " --> " + str(d_distances[word]))





#######################################################################################################
#######################################################################################################
if __name__ == '__main__':

    pathFile = "word2vecData_embeddings.json"
    d_w2vData = loadEmbeddings(pathFile)

    getCosSimilarityBetweenWords("substance", "dna", d_w2vData)

    getNearearWords("dna", d_w2vData, 0.9)




