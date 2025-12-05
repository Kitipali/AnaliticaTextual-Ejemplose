# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Manejo de expresiones regulares para extraer información simple
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

import regex


def BusquedaSimple(texto):
    r = regex.findall('[0-9]+',texto)
    return(r)


def ExtraerNombresDocumento(Lineas):
   lista = []
   patron = r'(\w+)/(PROPN|NOUN)'
   for linea in Lineas:
      ListaPalabras  = ExtraerNombresLinea(patron,linea)
      if ListaPalabras != []:
         lista.append(ListaPalabras)
   return(lista)

def ExtraerNombresLinea(patron,linea):
      texto      = linea.rstrip()
      etiquetado = Etiquetar(texto)    
      ListaPalabras   = [w for (w,t) in regex.findall(patron,etiquetado)]
      return(ListaPalabras)
    

def ExtraerNumeros(lineas):
    for linea in lineas:
      linea = linea.rstrip()
      match = regex.findall("[0-9]+", linea)
      if match:
           return(match)
      return(None)
  
    
def BusquedaRelacion(texto):
   etiquetado = Etiquetar(texto)
   patron =r'(\s*(\w+)/PROPN)+\s*\,/PUNCT\s*(\w+)/NOUN\s*\w+/ADP\s*(\w+)/PROPN'
   match = regex.search(patron, etiquetado)
   return(match)

   
FILENAME = 'E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d1.txt'

texto = "El numéro de contagiados pasó de 1000 a 2000 en una semana"
lista = BusquedaSimple(texto)
print(lista)

nlp = es_core_news_sm.load()
    
lineas = open(FILENAME)
Nombres = ExtraerNombresDocumento(lineas)
print(Nombres)

rel = BusquedaRelacion("Maria Barrientos Tapia, asesora del Ministerio")
if rel is not None:
    print(rel.captures(2))
    print(rel.captures(3))
    print(rel.captures(4))
