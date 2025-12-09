# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Indexación de documentos mediante vectores de peso (TFxIDF)
#             Ejemplo con motor de búsqueda de documentos
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine


def crearQuery(terms,idf,vocabulario):
    query = np.zeros(len(vocabulario))
    listaTerminos = Tokenizar(Lematizar(terms))
    for t in listaTerminos:      
       try:
           indice = vocabulario[t]
           query[indice] = 1
       except KeyError:
           indice = -1
    if (np.count_nonzero(query) != 0):
              query = query * idf
              return(query)
    return([])



def RecuperarDocumentosRelevantes(query,modelo,doc_id):
  RelDocs = []
  for ind_doc in range(len(doc_id)):
    filename = doc_id[ind_doc]  
    similitud = 1 - cosine(query,modelo[ind_doc,:])
    RelDocs.append((similitud,filename))  
  return(sorted(RelDocs,reverse=True))


def MostrarDocumentos(Docs):
    print("Lista de documentos relevantes a la query:\n")
    for (sim,d) in Docs:
        print("Doc: "+d+" ("+str(sim)+")\n")
    


PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"

nlp           = es_core_news_sm.load()
corpus,docsID = CrearCorpus(PATH)
textos  = PreProcesar(corpus)
CrearVSM(textos,"mi_modelo")

(tfidf, idf, vocabulario) = CargarModelo("mi_modelo")



print("*********************************************")
print("Bienvenido al buscador de documentos!")
print("*********************************************")

terms = input("Ingrese query: ")
vector_query = crearQuery(terms,idf,vocabulario)

if len(vector_query)==0:
    print("ERROR en vector de consulta, no se pueden recuperar documentos!..")
else:
    DocsRelevantes = RecuperarDocumentosRelevantes(vector_query,tfidf,docsID)
    MostrarDocumentos(DocsRelevantes)