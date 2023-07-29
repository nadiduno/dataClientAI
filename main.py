# Projeto Python IA: Inteligência Artificial e Previsões

### Case: Score de Crédito dos Clientes

#Você foi contratado por um banco para conseguir definir o score de crédito dos clientes. Você precisa analisar todos os clientes do banco e, com base nessa análise, criar um modelo que consiga ler as informações do cliente e dizer automaticamente o score de crédito dele: Ruim, Ok, Bom

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

tabela = pd.read_csv('https://raw.githubusercontent.com/nadiduno/dataClientAI/main/clientes.csv')
print(tabela.info())

#Modificar dados a códigos em lugar de string
codificador = LabelEncoder()
tabela['profissao'] = codificador.fit_transform(tabela['profissao'])
tabela['mix_credito'] = codificador.fit_transform(tabela['mix_credito'])
tabela['comportamento_pagamento'] = codificador.fit_transform(tabela['comportamento_pagamento'])
print(tabela.info())

#Previsão
#X(Columnas que desejo usar para fazer a previsão)
#Y(Que prever sscore_credito)
y=tabela['score_credito']
x=tabela.drop(['id_cliente', 'score_credito'],axis=1)

#Seprar os dados de treino e do teste

x_treino, x_teste, y_treino, y_teste =train_test_split(x,y)

#Criar modelo de AI
modelo_arvordecisao =RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

#Trenar o modelo 
modelo_arvordecisao.fit(x_treino,y_treino)
modelo_knn.fit(x_treino,y_treino)

#Acurácia do modelo (quantas veces acertou/qtd clientes)
#Acertos
previsao_arvoredecisao = modelo_arvordecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

print('Acurácia do Modelo de Arvore de Decisão',accuracy_score(y_teste,previsao_arvoredecisao))
print('Acurácia do Modelo KNN',accuracy_score(y_teste,previsao_knn))
