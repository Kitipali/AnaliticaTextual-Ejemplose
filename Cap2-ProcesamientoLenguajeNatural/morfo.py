# ***********************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Funciones para realizar análisis morfológico del tipo 
#             Stemming o Lematización
# ***********************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from nltk.stem import SnowballStemmer
import es_core_news_sm
from spacy.lang.es.stop_words import STOP_WORDS
from string import punctuation
import regex



def LeerTexto(FileName):
    f = open(FileName, 'r')
    texto = f.read().split('.')
    f.close()
    return(texto)


def PreProcesamiento(oraciones):
    texto_limpio = []
    for texto in oraciones:  
       texto = EliminarStopwords(texto)    
       texto = EliminaNumeroYPuntuacion(texto)      
       if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          texto_limpio.append(texto)
    return(texto_limpio)



def Reducir(oracion): 
   tokens = Tokenizar(oracion)
   stems = [stemmer.stem(palabra) for palabra in tokens]
   return(Lista_a_Oracion(stems)) 



FILENAME='E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d1.txt'

nlp       = es_core_news_sm.load()
stemmer   = SnowballStemmer('spanish')  
oraciones_originales = LeerTexto(FILENAME)
oraciones = PreProcesamiento(oraciones_originales)

texto_lematizado  = [Lematizar(oracion) for oracion in oraciones]
texto_reducido    = [Reducir(oracion) for oracion in oraciones]
print(texto_lematizado)
print(texto_reducido)



