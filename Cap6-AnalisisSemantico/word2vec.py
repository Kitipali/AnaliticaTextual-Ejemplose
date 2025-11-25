# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Generación de word embeddings con Word2Vec
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity



def PreProcesarOraciones(textos):
    texto_limpio = []
    for texto in textos:  
        texto  = EliminarStopwords(texto)    
        texto  = Lematizar(texto)     
        texto  = EliminaNumeroYPuntuacion(texto) 
        if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          tokens = Tokenizar(texto)
          texto_limpio.append(tokens)
    return(texto_limpio)

   
def EntrenarModelo(oraciones,NombreModelo):
    model = Word2Vec(oraciones, size=4, window=2, min_count=1)
    model.save(NombreModelo)

    
def CargarModelo(NombreModelo):
   modelo = Word2Vec.load(NombreModelo)
   vocabulario = [term for term in modelo.wv.vocab]  
   return(modelo,vocabulario)


    
PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"

nlp = es_core_news_sm.load()
corpus,_doc_id = CrearCorpus(PATH)
oraciones = PreProcesarOraciones(corpus)
EntrenarModelo(oraciones,'mi_word2vec')

modelo, vocabulario = CargarModelo('mi_word2vec')

print(modelo.wv.most_similar('sellar'))
print(modelo.wv.similarity("sellar","enrocar"))

v1 = modelo.wv['ganador']
v2 = modelo.wv['perder']
v3 = modelo.wv['ventaja']

modelo.wv.similar_by_vector(v1-v2+v3)

GraficarVectores(vocabulario,modelo.wv.vectors[0:20,0:2])


similitud = 1 - cosine_similarity(modelo.wv.vectors)
plt.matshow(similitud[0:20,0:20])
plt.show()
