# ***********************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Funciones para realizar parsing de dependencias en Español
# ***********************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')


import es_core_news_sm
from spacy import displacy
from pathlib import Path


def LeerTexto(FileName):
    f = open(FileName, 'r',encoding="latin-1")
    texto = f.read().split('.')
    f.close()
    return(texto)


def CrearArchivoSalida(ID_oracion,imagen):
    file_name = "File_"+ str(ID_oracion) + ".svg"
    output_path = Path(DIR_IMAGENES + file_name)
    output_path.open("w", encoding="utf-8").write(imagen)
    

FILENAME     ='E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d20.txt'
# Crear directorio en donde estarán las imágenes de parsing generadas
DIR_IMAGENES = 'c:/Users/atkin/Desktop/imagenes/'

nlp = es_core_news_sm.load()
oraciones = LeerTexto(FILENAME)

doc = nlp("Juanito compró un huevito para entregarlo a papa")
dep = [d.dep_ for d in doc]
print(dep)

ID_oracion = 0
for oracion in oraciones:
    doc = nlp(oracion)
    imagen = displacy.render(doc, style="dep", jupyter=False)
    CrearArchivoSalida(ID_oracion,imagen)
    ID_oracion +=1
    