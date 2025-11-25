# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Generación de espacios semánticos con LSA
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

 
def CrearEspacioLSA(corpus,numDim,NombreModelo):
  textos  = PreProcesar(corpus)
  transf = TfidfVectorizer()
  tf = transf.fit_transform(textos).T
  U, Sigma, VT = np.linalg.svd(tf.toarray())
  terms = np.dot(U[:,:numDim], np.diag(Sigma[:numDim]))
  docs = np.dot(np.diag(Sigma[:numDim]), VT[:numDim, :]).T 
  vocab = transf.get_feature_names()
  GrabarModeloLSA(NombreModelo, Sigma, terms, docs, vocab)


def GrabarModeloLSA(NombreModelo,Sigma,terms,docs, vocab):
   existe = os.path.isdir(NombreModelo)
   if not existe:
       os.mkdir(NombreModelo)
   joblib.dump(Sigma,   NombreModelo +"/"+'sigma.pkl') 
   joblib.dump(terms,   NombreModelo +"/"+'terms.pkl') 
   joblib.dump(docs,    NombreModelo +"/"+'docs.pkl') 
   joblib.dump(vocab,   NombreModelo +"/"+'vocab.pkl') 

    
def CargarModeloLSA(NombreModelo):
    sigma   = joblib.load(NombreModelo+"/"+'sigma.pkl')
    terms   = joblib.load(NombreModelo+"/"+'terms.pkl')
    docs    = joblib.load(NombreModelo+"/"+'docs.pkl')
    vocab   = joblib.load(NombreModelo+"/"+'vocab.pkl')
    return(sigma, terms, docs, vocab)



# ************************************************************
# Función: GraficarImportancia(Sigma)
# Objetivo: Graficar la importancia (cuadrado de valores singulares) 
#       sobre el numero de valores singulares (dimensiones) con el fin de elegir
#       el número adecuado de dimensiones
# ************************************************************  
    
def GraficarImportancia(Sigma):   
    NumValores = np.arange(len(Sigma))
    Importancia = [x**2 for x in Sigma]
    plt.bar(NumValores,Importancia)
    plt.ylabel('Importancia')
    plt.xlabel('Valores Singulares')
    plt.title('Importancia de Valores Singulares en SVD')
    plt.show()


def CrearDiccionario(Vectores,vocabulario):
   dicc = {}
   for  v in range(0,len(vocabulario)):
      dicc[vocabulario[v]] = Vectores[v]
   return(dicc)

# **************************************************************
# Comienzo programa principal
# **************************************************************
  
# Modificar la ruta PATH con la ubicación de un corpus a utilizar
PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"

nlp = es_core_news_sm.load()
NumDim =  15
corpus, lista_docs = CrearCorpus(PATH)

CrearEspacioLSA(corpus,NumDim,"mi_lsa")

(Sigma, vect_terms, vect_docs, lista_palabras) = CargarModeloLSA("mi_lsa")

GraficarImportancia(Sigma)   



GraficarVectores(lista_palabras, vect_terms[0:20])
GraficarVectores(lista_docs    , vect_docs[0:10])


terms = CrearDiccionario(vect_terms, lista_palabras)
docs  = CrearDiccionario(vect_docs , lista_docs)





