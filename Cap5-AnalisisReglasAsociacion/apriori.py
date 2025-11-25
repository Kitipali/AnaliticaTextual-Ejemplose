# ****************************************************************************
# Author      : John Atkinson
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Extracción de reglas de asociación desde documentos utilizando
#              algoritmo APRIORI
#*****************************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


from mlxtend.frequent_patterns import apriori, association_rules 


  
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
          texto_limpio.append(texto)
    return(texto_limpio)


def ExtraerNombresLinea(patron,linea):
      texto      = linea.rstrip()
      etiquetado = Etiquetar(texto)    
      ListaPalabras   = [w for (w,t) in regex.findall(patron,etiquetado)]
      return(ListaPalabras)
  
    
      
PATH = "E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/"

# Crear corpus con textos de directorio apuntado por PATH 


nlp = es_core_news_sm.load()

corpus,_ = CrearCorpus(PATH)
textos = PreProcesarConNombres(corpus)
CrearVSM(textos,"transacciones",False,True)

(doc_binarios,_,vocabulario) = CargarModelo("transacciones")
    

# Para agilizar la extracción, elegir los 30 primeros términos por ahora
# De lo contrario:   LargoVoc = doc_binario.shape[1]
LargoVocab  = 30 

canasta = doc_binarios[:,0:LargoVocab]

NombreColumnas = [ list(vocabulario)[x]  for x in range(0,LargoVocab)]

df = pd.DataFrame(canasta, columns = NombreColumnas)


itemsets = apriori(df, min_support = 0.005, use_colnames = True)
  
reglas = association_rules(itemsets, metric ="lift", min_threshold = 1) 
reglas_ordenadas = reglas.sort_values(['confidence', 'lift'], 
                                      ascending =[False, False]) 

print(reglas_ordenadas[['antecedents','consequents','confidence','lift']])
