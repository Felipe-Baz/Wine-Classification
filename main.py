#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
arq = pd.read_csv('C:/Users/felip/Desktop/machine learning/data/wine_dataset.csv')

arq.head()

arq['style'] = arq['style'].replace('red', 0)

arq['style'] = arq['style'].replace('white', 1)

arq.head(1000000)

y = arq['style']
x = arq.drop('style', axis=1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print(resultado)

y_teste[410:420]

x_teste[410:420]

previsoes = modelo.predict(x_teste[410:420])

print(previsoes)

