# ****************************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             - Varias funciones utilizadas en los ejemplos del libro
#             - Este módulo se carga desde los otros programas de ejemplo
#*****************************************************************************

import numpy as np
import pandas as pd
import os
import regex
import es_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from spacy.lang.es.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
import joblib

def ConvertirAcentos(texto):
    texto=texto.replace("\xc3\xa1","á")
    texto=texto.replace("\xc3\xa9","é")   
    texto=texto.replace("\xc3\xad","í")
    texto=texto.replace("\xc3\xb3","ó")
    texto=texto.replace("\xc3\xba","ú")
    texto=texto.replace("\xc3\x81","Á")
    texto=texto.replace("\xc3\x89","É")
    texto=texto.replace("\xc3\x8d","Í")
    texto=texto.replace("\xc3\x93","Ó")
    texto=texto.replace("\xc3\x9a","Ú")
    texto=texto.replace("\xc3±","ñ")
    return(texto)

def CrearCorpus(path):
  directorio = os.listdir(path)
  corpus = []
  doc_id = []  
  for filename  in directorio:
     texto = open(path+filename,'r',encoding="latin-1").read()
     texto = ConvertirAcentos(texto)
     corpus.append(texto)
     doc_id.append(filename)
  return(corpus,doc_id)

   
def PreProcesar(textos):
    texto_limpio = []
    for texto in textos:  
        texto = EliminarStopwords(texto.lower())    
        texto = Lematizar(texto)     
        texto = EliminaNumeroYPuntuacion(texto)      
        if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          texto_limpio.append(texto)
    return(texto_limpio)


def CargarModeloLSA(NombreModelo):
    sigma   = joblib.load(NombreModelo+"/"+'sigma.pkl')
    terms   = joblib.load(NombreModelo+"/"+'terms.pkl')
    docs    = joblib.load(NombreModelo+"/"+'docs.pkl')
    vocab   = joblib.load(NombreModelo+"/"+'vocab.pkl')
    return(sigma, terms, docs, vocab)


def CrearVSM(textos,nombre_modelo,modelo_idf=True,modelo_binario=False):
  modelo  = TfidfVectorizer(use_idf=modelo_idf, 
                            norm=None, binary=modelo_binario)
  matriz_features  = modelo.fit_transform(textos)
  vocabulario      = modelo.vocabulary_
  dtm              = matriz_features.toarray()
  if (modelo_idf):
        idf = modelo.idf_
  else: 
        idf = []   
  GrabarModelo(nombre_modelo,dtm,idf,vocabulario)
  
  
def GrabarModelo(NombreModelo,modelo,idf,vocab):
   existe = os.path.isdir(NombreModelo)
   if not existe:
       os.mkdir(NombreModelo)
   joblib.dump(modelo,   NombreModelo +"/"+'tfidf.pkl') 
   joblib.dump(idf,   NombreModelo +"/"+'idf.pkl') 
   joblib.dump(vocab, NombreModelo +"/"+'vocab.pkl') 

   
    
def CargarModelo(NombreModelo):
    modelo = joblib.load(NombreModelo+"/"+'tfidf.pkl')
    idf   = joblib.load(NombreModelo+"/"+'idf.pkl')
    vocab  = joblib.load(NombreModelo+"/"+'vocab.pkl')
    return(modelo,idf,vocab)


def EliminaNumeroYPuntuacion(oracion):
    string_numeros = regex.sub(r'[\”\“\¿\°\d+]','', oracion)
    return ''.join(c for c in string_numeros if c not in punctuation)


def Lematizar(oracion):
   doc = nlp(oracion)
   lemas = [token.lemma_ for token in doc]
   return(Lista_a_Oracion(lemas))  
  
def Lista_a_Oracion(Lista):
   return(" ".join(Lista))          

def EliminarStopwords(oracion):
    Tokens = Tokenizar(oracion)
    oracion_filtrada =[] 
    for palabra in Tokens:
       if palabra not in STOP_WORDS:
           palabra_limpia = palabra.rstrip()
           if len(palabra_limpia)!=0:
              oracion_filtrada.append(palabra_limpia) 
    return(Lista_a_Oracion(oracion_filtrada))

def Tokenizar(oracion):
    doc = nlp(oracion)
    tokens = [palabra.text for palabra in doc]
    return(tokens)


def Etiquetar(texto):
   doc = nlp(texto)
   Etiquetado = ''.join(t.text+"/"+t.pos_+" " for t in doc)
   return(Etiquetado.rstrip())


def GraficarVectores(vocab,vectores):
    x = []
    y = []
    for value in vectores:
        x.append(value[0])
        y.append(value[1])   
    plt.figure(figsize=(7, 7))   
    plt.title("Representación de Vectores")
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(vocab[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()