# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Generación de modelos de tópicos con LDA
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim as gensimvis


def PreProcesarConNombres(textos):
    texto_limpio = []
    patron = r'(\w+)/(PROPN|NOUN)'
    for texto in textos:  
        texto = Lista_a_Oracion(ExtraerNombresLinea(patron,texto))
        texto = EliminarStopwords(texto)    
        texto = Lematizar(texto)     
        texto = EliminaNumeroYPuntuacion(texto)      
        if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          tokens = Tokenizar(texto)
          texto_limpio.append(tokens)
    return(texto_limpio)


def ExtraerNombresLinea(patron,linea):
      texto      = linea.rstrip()
      etiquetado = Etiquetar(texto)    
      ListaPalabras   = [w for (w,t) in regex.findall(patron,etiquetado)]
      return(ListaPalabras)
  
      
def VisualizarLDA(modelo,corpus,diccionario):
   vis_data = gensimvis.prepare(modelo, corpus, diccionario)
   pyLDAvis.show(vis_data)
    


def MostrarEvolucionMetrica(corpus,dicc,MaxDoc,metrica="perp"):
    descrip = {"perp":"Perplejidad",
               "u_mass":"Coherencia"}
    score = []   
    list_k = list(range(2, MaxDoc))  
    for K in list_k:
       modelo = gensim.models.ldamodel.LdaModel(corpus, num_topics=K, 
                                                     id2word = dicc)
       if (metrica=="perp"):
          valor = modelo.log_perplexity(corpus)
       else:
          coherence = CoherenceModel(model=modelo, corpus=corpus, 
                                     dictionary=dicc, coherence="u_mass")
          valor = coherence.get_coherence()
       score.append(valor)
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, score, '-o')
    plt.xlabel(r'Número de Tópicos (K)')
    plt.ylabel(descrip[metrica])

  

PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"


nlp = es_core_news_sm.load()

texts, _doc_id   = CrearCorpus(PATH)
texts  = PreProcesarConNombres(texts)
MaxDoc = 10
K=3



diccionario = corpora.Dictionary(texts) 
corpus = [diccionario.doc2bow(text) for text in texts]

modeloLDA = gensim.models.ldamodel.LdaModel(corpus, num_topics=K, 
                                           id2word = diccionario)

print(modeloLDA.print_topics(num_topics=K, num_words=4))


MostrarEvolucionMetrica(corpus,diccionario,MaxDoc,"perp")
MostrarEvolucionMetrica(corpus,diccionario,MaxDoc,"u_mass")

VisualizarLDA(modeloLDA,corpus,diccionario)
