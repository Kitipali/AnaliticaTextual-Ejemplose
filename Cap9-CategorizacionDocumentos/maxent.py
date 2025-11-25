# ****************************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Categorización de sentimientos utilizando clasificador
#                Máxima Entropía (MaxEnt)
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import MaxentClassifier
from nltk.probability import FreqDist


def CrearDiccionario(Tokens_clase1,Tokens_clase2,MaxTerms):
   ListaPalabras = Tokens_clase1 + Tokens_clase2
   FreqPalabras = FreqDist(ListaPalabras)
   claves        = list(FreqPalabras.keys())
   return (set(claves[:MaxTerms]))


def CrearConjuntoFeatures(Tokens_clase1,Tokens_clase2): 
  Features_clase1 = [({t:True},Clase1) for t in Tokens_clase1]
  Features_clase2 = [({t:True},Clase2) for t in Tokens_clase2]
  return((Features_clase1,Features_clase2))


def CrearTrainingFeatures(Features_clase1,Features_clase2,PropTraining):
  (clase1cutoff,clase2cutoff) = ObtenerCutoff(Features_clase1,
                                              Features_clase2,
                                              PropTraining)
  return (Features_clase1[:clase1cutoff]+Features_clase2[:clase2cutoff])


def CrearTestingFeatures(Features_clase1,Features_clase2,PropTraining):
  (clase1cutoff,clase2cutoff) = ObtenerCutoff(Features_clase1,
                                              Features_clase2,
                                              PropTraining)
  test_clase1 = int(len(Features_clase1)-clase1cutoff)
  test_clase2 = int(len(Features_clase2)-clase2cutoff)
  return(Features_clase1[-test_clase1:] + Features_clase2[-test_clase2:])


def ObtenerCutoff(Features_clase1,Features_clase2,PropTraining):
  clase1cutoff = int(len(Features_clase1)*PropTraining)
  clase2cutoff = int(len(Features_clase2)*PropTraining)
  return(clase1cutoff,clase2cutoff)   
    
   
# ***************************************************************************
# Eliga el algoritmo GIS (algoritmo(0)) como escalamiento iterativo para obtener pesos
# Posibles métodos: ['GIS', 'IIS', 'MEGAM', 'TADM']
# Puede clasificar con Naive Bayes también.
#  Para esto: model = NaiveBayesClassifier.train(trainingFeatures)
# ***************************************************************************

def EntrenarModelo(trainingFeatures):
   model = MaxentClassifier.train(trainingFeatures, 
                                  algorithm="GIS",max_iter=4,trace=0)
   return(model)

   
def PreProcesarConNombres(textos):
    texto_limpio = []
    patron = r'(\w+)/(PROPN|NOUN)'   
    for texto in textos:  
        texto = Lista_a_Oracion(ExtraerNombresLinea(patron,texto))
        texto = EliminarStopwords(texto.lower())    
        texto = Lematizar(texto)     
        texto = EliminaNumeroYPuntuacion(texto)
        if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          texto_limpio.append(texto)
    return(texto_limpio)

def ExtraerNombresLinea(patron,linea):
      texto      = linea.rstrip()
      etiquetado = Etiquetar(texto)    
      ListaPalabras   = [w for (w,t) in regex.findall(patron,etiquetado)]
      return(ListaPalabras)
  
def TokenizarDocumentos(textos):
    lista = []
    for t in textos:
        tokens = Tokenizar(t)
        for tok in tokens:
            lista.append(tok)
    return(lista)
         

def CrearFeatures(texto):
    l = Lematizar(texto.lower())
    t = Tokenizar(l)
    f = GenerarFeaturesPalabras(t)
    return(f)

def GenerarFeaturesPalabras(words):
    return {word:True for word in words if word in top_words}
   



PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/"

nlp = es_core_news_sm.load()
Clase1 = "deportes"
Clase2 = "coronavirus"

corpus_clase1,_  = CrearCorpus(PATH+Clase1+"/")
corpus_clase2,_  = CrearCorpus(PATH+Clase2+"/")

corpus_clase1 = PreProcesarConNombres(corpus_clase1)
corpus_clase2 = PreProcesarConNombres(corpus_clase2)

Tokens_clase1 = TokenizarDocumentos(corpus_clase1)
Tokens_clase2 = TokenizarDocumentos(corpus_clase2)


MAXTERMS = 1000
top_words = CrearDiccionario(Tokens_clase1,Tokens_clase2,MAXTERMS)
(Features_clase1,Features_clase2) = CrearConjuntoFeatures(Tokens_clase1,
                                                          Tokens_clase2)


training_features = CrearTrainingFeatures(Features_clase1,Features_clase2,0.7)
testing_features  = CrearTestingFeatures(Features_clase1,Features_clase2,0.7)

Clasificador = EntrenarModelo(training_features)

# Mostrar los 3 features mas informativos|

Clasificador.show_most_informative_features(10)


print("Exactitud de Training: ")
print(nltk.classify.accuracy(Clasificador, training_features))
print("Exactitud de Testing: ")
print(nltk.classify.accuracy(Clasificador, testing_features))

# Clasificar por MaxEnt dado un texto de entrada (features)

categoria = Clasificador.classify(
                      CrearFeatures("víctimas de la pandemia en el mundo"))

print("Categoría del texto: "+categoria)


