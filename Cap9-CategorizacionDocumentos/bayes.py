# ****************************************************************************
# Author      : John Atkinson      (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Categorización de sentimientos utilizando clasificador
#                Naive Bayes
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')



from textblob.classifiers import NaiveBayesClassifier 
from numpy import unique
import csv
from pylab import *


def CargarArchivoCSV(NombreArchivo, EsTextoNuevo=False):
    fileID = open(NombreArchivo)
    texto = csv.reader(fileID, delimiter=',')
    if (EsTextoNuevo):
       Opiniones = [(oracion,fecha.strip()) for (oracion,fecha) in texto]      
    else:
       Opiniones = [(oracion,senti.strip()) for (oracion,senti) in texto]
    return(Opiniones)


def ClasificarOpiniones(modelo,ListaPruebas):
    ListaSenti = [ (modelo.classify(orac),fecha) 
                           for (orac,fecha) in ListaPruebas]
    ListaCont = ContarOpiniones(ListaSenti)
    return(ListaCont)

def ContarOpiniones(ListaSenti):
    ListaFechas = unique([fecha  for (sent,fecha) in ListaSenti])
    ListaFrecuencias = []
    for Fecha in ListaFechas:
      cuentaPOS = 0
      cuentaNEG = 0
      for (s,f) in ListaSenti:
        if (f== Fecha):
            if (s == "neg"):
                cuentaNEG += 1 
            else:
                cuentaPOS += 1
      ListaFrecuencias.append((Fecha,cuentaPOS,cuentaNEG))
    return(ListaFrecuencias)


def GraficarEvolucionSentimientos(ListaFechas,ListaCantPOS,ListaCantNEG):
   ax = figure().gca()
   ax.plot(ListaFechas,ListaCantPOS,"o-" "r",ListaFechas,ListaCantNEG,"o-") 
   ax.yaxis.set_major_locator(MaxNLocator(integer=True)) 
   title("Evolución de Sentimientos")
   legend(('Positivas', 'Negativas'), loc = 'upper right')
   xlabel('Fecha')
   ylabel('Núm. Opiniones')
   show()  

def LematizarTextos(ListaOraciones):
    ListaLema = []
    for (oracion, senti) in ListaOraciones:  
       OracionLematizada = Lematizar(oracion)
       ListaLema.append((OracionLematizada,senti))
    return(ListaLema)
   

PATH = "./"


nlp = es_core_news_sm.load()

training    = CargarArchivoCSV(PATH+"training.csv")
testing     = CargarArchivoCSV(PATH+"testing.csv")
#training = LematizarTextos(training)
#testing  = LematizarTextos(testing)


modelo = NaiveBayesClassifier(training)
TextosNuevos   = CargarArchivoCSV(PATH+'nuevos.csv',True)
ListaContFechas = ClasificarOpiniones(modelo,TextosNuevos)


Fechas = [f for (f,_cpos,_cneg) in ListaContFechas]
CantPOS = [cpos for (_f,cpos,_cneg) in ListaContFechas]
CantNEG = [cneg for (_f,_cpos,cneg) in ListaContFechas]

GraficarEvolucionSentimientos(Fechas,CantPOS,CantNEG)


accTrain = round(modelo.accuracy(training),2)
accTest  = round(modelo.accuracy(testing),2)
print(accTrain)
print(accTest)



# Probabilidad que una oración sea de clase POSITIVA
# orac ="Me gusta el servicio de banda ancha"
# prob = modelo.prob_classify(Lematizar(orac))
# print(round(prob.prob("pos"),2))
