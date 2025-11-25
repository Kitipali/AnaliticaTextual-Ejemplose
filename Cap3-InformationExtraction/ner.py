# ***********************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Reconocimiento de entidades nombradas (NER)
#             Visualización de las entidades identificadas en un texto
#             TAGSET:  https://spacy.io/api/annotation#pos-tagging
#
# INSTALAR el paquete:  pip install tika
#**************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

import es_core_news_sm
from spacy import displacy
          
           
def ExtraerEntidades(texto):
   doc = nlp(texto) 
   entities = [NE for NE in doc.ents]
   return(entities)


def FiltrarEntidades(Entidades, tipo_entidad):
   entidades = list()
   for Ent in Entidades:
        if (Ent.label_ == tipo_entidad):
            entidades.append(Ent.text)
   return(entidades)


FILENAME = 'E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d20.txt'

nlp = es_core_news_sm.load()

texto = open(FILENAME, 'r',encoding="utf-8").read()
texto.rstrip("\n")

entidades= ExtraerEntidades(texto)
print(entidades)
entidadesTipo = FiltrarEntidades(entidades,'ORG')
print(entidadesTipo)


# La visualización se puede observar en: http://localhost:5000/

doc = nlp(texto)
displacy.serve(doc, style="ent")

