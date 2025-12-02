# ***********************************************************************
# Author      : John Atkinson    (atkinsonabutridy@gmail.com)
# Book        : Analítica Textual
# Last Updated: 02-July-2020
# Contents: 
#             Etiquetado léxico del tipo Part-of-Speech (POS)
# ***********************************************************************

from past.builtins import execfile
execfile('E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/utils.py')

import es_core_news_sm


 
FILENAME='E:/JOHN/BOOK/SPANISH/TextAnalytics-Examples/CORPUS/deportes/d20.txt'
   
nlp = es_core_news_sm.load()

texto = open(FILENAME, 'r',encoding="utf-8").read()
texto_etiquetado = Etiquetar(texto)
print(texto_etiquetado)
