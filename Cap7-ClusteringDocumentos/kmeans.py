# ****************************************************************************
# Author      : John Atkinson     (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Clustering de documentos con K-Means
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS


def clustering(MatrizDatos,K):
  modelo = KMeans(n_clusters=K)   
  modelo.fit(MatrizDatos)        
  return(modelo)


def Escalar(data_matrix):
    dist = 1 - cosine_similarity(data_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed")
    pos = mds.fit_transform(dist)  
    x, y = pos[:, 0], pos[:, 1]
    return(x,y)


def VisualizarClusters(clusters,titles,MatrizDatos):
    (x_nueva,y_nueva) = Escalar(MatrizDatos)
    color_cluster = {0: '#1b9e77', 1: '#d95f02', 
                      2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    nombre_cluster = {0: 'Cluster1', 
                 1: 'Cluster2', 
                 2: 'Cluster3', 
                 3: 'Cluster4', 
                 4: 'Cluster5'}
    df = pd.DataFrame(dict(x=x_nueva,y=y_nueva,label=clusters,title=titles)) 
    grupos = df.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9)) 
    for nombre, grupo in grupos:
        ax.plot(grupo.x, grupo.y, marker='o', linestyle='', 
               label=nombre_cluster[nombre], color=color_cluster[nombre]) 
    ax.legend() 
    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  
    plt.show() 


def NormalizarMetrica(scores):
    maximo = np.max(scores)
    nuevo_valor = [x/maximo for x in scores]   
    return(nuevo_valor)

    
def MostrarEvolucionMetrica(Datos,MaxDoc,metrica="sse"):
    descripcionMetrica = {"sse":"Suma de distancias cuadradas",
                          "sil":"Coeficiente Silhouette"}
    score = []  
    list_k = list(range(2, MaxDoc))  
    for k in list_k:
       Clusters = clustering(Datos,K)
       if  metrica =="sse":
           valor = Clusters.inertia_
       else:
           valor = silhouette_score(Datos, Clusters.labels_)
       score.append(valor)   
    score = NormalizarMetrica(score)   
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, score, '-o')
    plt.xticks(list_k)
    plt.xlabel(r'Número de clusters (K)')
    plt.ylabel(descripcionMetrica[metrica])
    
    
PATH = "E:/JOHN/BOOK/TextAnalytics-Examples/CORPUS/deportes/"


nlp = es_core_news_sm.load()

(corpus,docID) = CrearCorpus(PATH)
textos  = PreProcesar(corpus)

CrearVSM(textos,"mi_tfidf")


(vectores, _idf, vocabulario) = CargarModelo("mi_tfidf")

K = 4    
modelo = clustering(vectores,K)
print(modelo.labels_)

VisualizarClusters(modelo.labels_,docID,vectores)

#MaxDoc = len(docID)

MaxDoc = 10

MostrarEvolucionMetrica(vectores,MaxDoc,"sse")
MostrarEvolucionMetrica(vectores,MaxDoc,"sil")

# Para modelo de vectores LSA, Ud. debe haber generado previamente 
# el modelo "mi_lsa"

# Ubicación donde se encuentra el modelo LSA generado:
dirLSA = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/Cap6-AnalisisSemantico/"
(_Sigma, _terminos, vectores, _vocab) = CargarModeloLSA(dirLSA + "mi_lsa")

modelo = clustering(vectores,K)
print(modelo.labels_)

VisualizarClusters(modelo.labels_,docID,vectores)

MostrarEvolucionMetrica(vectores,MaxDoc,"sse")
MostrarEvolucionMetrica(vectores,MaxDoc,"sil")