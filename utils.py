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
import es_core_news_sm # carga el modelo en español de Spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from spacy.lang.es.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
import joblib
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("spanish")

nlp = es_core_news_sm.load()
"""
Te da un Doc, que contiene:

doc[0] → primer token

doc[0].text → texto

doc[0].lemma_ → lema

doc[0].pos_ → categoría gramatical

doc[0].dep_ → dependencia

doc.ents → entidades del texto

doc.sents → oraciones

etc.

Es literalmente EL corazón de spaCy.
"""
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
    string_numeros = regex.sub(r'[\”\“\¿\°\d]','', oracion)
    return ''.join(c for c in string_numeros if c not in punctuation) 
    #Recorre cada carácter c de string_numeros y se queda SOLO con los que NO estén en punctuation.
    #como ese recorrido es por caracter, es necesario juntarlos (join) sin espacios "", ya que los propios espacios no se han borrado porque no están en la lusta punctuation 

"""
Esta expresión regular busca dentro del texto:

\d → cualquier número (0–9)

\” → comillas especiales (Unicode)

\“ → comillas de apertura

\¿ → signo de interrogación inicial

\° → símbolo de grados


Todo lo que coincide con eso lo reemplaza por '' (carácter nulo).
"""


def Lematizar(oracion): #Consiste en convertir cada palabra a su forma base o lema. Corriendo ->correr, habló -> hablar, fue -> ser
   doc = nlp(oracion)  # Pasa este exto por el pipeline del modelo spacy en español. En spacy no hace falyta tokenizar previamnete
   lemas = [token.lemma_ for token in doc] # Esto es una list comprehension que recorre cada token del documento y lo convierte en su forma base (lema)
   return(Lista_a_Oracion(lemas))   # para VOLVER a convertir esa lista en una cadena.

def Reducir(oracion):
    tokens = Tokenizar(oracion)
    stems = [stemmer.stem(palabra) for palabra in tokens] #aquí usa snowball -> necesita tokenizar previamnete
    return Lista_a_Oracion(stems)
  
def Lista_a_Oracion(Lista): #Convierte una lista de palabras en una oración uniendo cada token con un espacio.
   return(" ".join(Lista))          

def EliminarStopwords(oracion):
    Tokens = Tokenizar(oracion)
    oracion_filtrada =[] 
    for palabra in Tokens:
       if palabra not in STOP_WORDS:
           palabra_limpia = palabra.rstrip() #rstrip() ->elimina espacios en blanco al final de la palabra.
           if len(palabra_limpia)!=0:
              oracion_filtrada.append(palabra_limpia) 
    return(Lista_a_Oracion(oracion_filtrada))

def Tokenizar(oracion):
    doc = nlp(oracion) # Pasa este exto por el modelo spacy en español
    tokens = [palabra.text for palabra in doc] #Recorre cada token (palabra) dentro del documento spaCy (doc), extrae el atributo .text (su contenido original).
    return(tokens)


def Etiquetar(texto):
   doc = nlp(texto) # genera una lista de objetos Token -> varios atributos. Entre ellos .text(el texto) y .pos_(Categoría gramatical)
   Etiquetado = ''.join(t.text+"/"+t.pos_+" " for t in doc) #generamos un string texto de cada token + / + su categoría gramatical. Este string termina con un espacio extra al final
   return(Etiquetado.rstrip()) #.rstrip elimia ese espacio extra del final
"""
La salida sería de este tipo -> El/DET perro/NOUN corre/VERB rápido/ADV ./PUNCT

"""

def Etiquetar2(texto):
    doc = nlp(texto) # genera una lista de objetos Token -> varios atributos. Entre ellos .text(el texto) y .pos_(Categoría gramatical)
    etiquetado = [(token.text, token.pos_) for token in doc] #genera una lista de tuplas, donde cada elemento es (token.text, token.pos_) 
    return etiquetado

"""
La salida sería de este tipo -> Devuelve una lista de tuplas:
[
 ('El', 'DET'),
 ('perro', 'NOUN'),
 ('corre', 'VERB'),
 ('rápido', 'ADV'),
 ('.', 'PUNCT')
]

"""
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