# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Clustering de documentos con Mapas Auto-Organizativos (SOM)
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

import SimpSOM as sps

def EntrenarSOM(data,epocs=500,learningRate=0.01):
    mapa = sps.somNet(10, 10, data)
    mapa.train(learningRate, epocs)
    mapa.save("mi_som")
    return(mapa)


PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"

nlp = es_core_news_sm.load()

(corpus,docID) = CrearCorpus(PATH)
textos  = PreProcesar(corpus)
CrearVSM(textos,"mi_tf")

(tfidf, _idf, vocabulario) = CargarModelo("mi_tf")

datos = tfidf[:,0:6]

mapa=EntrenarSOM(datos)

proyectado = mapa.project(datos,labels=docID,show=True, colnum=0)
proyectado = mapa.project(datos,show=True,colnum=0)
clusters= mapa.cluster(datos,show=True)
print(clusters)

