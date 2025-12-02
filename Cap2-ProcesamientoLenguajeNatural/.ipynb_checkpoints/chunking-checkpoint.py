# ***********************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Funciones para realizar chunking (parsing parcial) de
#             textos en Español
# ***********************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from nltk.chunk import RegexpParser as RE_Parser
import es_core_news_sm



def Chunk_FN(texto):
   doc = nlp(texto)
   lista_FN = [chunk.text for chunk in doc.noun_chunks]
   return(lista_FN)    


def Chunk(texto):
  tagged1 = Etiquetar(texto). split()
  tagged = [tuple(s.split('/')) for s in tagged1]
  print(tagged)
  # Definir un chunk del tipo Frase Nominal (FN). 
  # Ejemplo: articulo* nombre* adjetivo*
  gramatica = '''                                                                                                              
    FN:                                                                                                                    
        {<DET>*(<PROPN|NOUN>)+<ADJ>*}
    '''
  chunker = RE_Parser(gramatica)
  Arbol = chunker.parse(tagged)
  matches =[]
  for subarbol in Arbol.subtrees():
        if subarbol.label() == 'FN': 
            matches.append(subarbol)
  return(matches)


def Subarbol_a_Lista(Chunks):
  lista =[]
  for c in Chunks:
    palabras = []
    for (palabra,_pos) in c:
        palabras.append(palabra)
    lista.append(' '.join(palabras))
  return(lista)  



FILENAME='E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d1.txt'


nlp = es_core_news_sm.load()

texto = open(FILENAME, 'r').read()
FN = Chunk_FN(texto)
print(FN)
FN2 = Subarbol_a_Lista(Chunk(texto))
print(FN2)
    

