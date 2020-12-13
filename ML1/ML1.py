import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random

from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')

data = pd.read_csv('diabetes_dataset.csv')
data_app = pd.read_csv('diabetes_app.csv')
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI", "DiabetesPedigreeFunction", "Age"]
powers = [0.4, 0.8, 0.6, 0.3, 0.75, 0.5, 0.5, 0.5]

def read():
    global data
    global data_app
    global columns
    global powers
    data = pd.read_csv('diabetes_dataset.csv')
    data_app = pd.read_csv('diabetes_app.csv')

def clear():
    global data
    global data_app
    global columns
    global powers

    powers.clear()
    columns.clear()
    columns = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    powers = [1, 1, 3, 3, 1, 1]
    data.pop('Insulin')
    data.pop('SkinThickness')
    data_app.pop('Insulin')
    data_app.pop('SkinThickness')

def estimate (i,j):
    global data
    global data_app
    global columns
    global powers

    distance = 0.0
    average = 0.0
    nb = [[],[]]
    for k in range(len(data[columns[0]])):
        distance = 0.0
        if (data.iloc[j, 6] == data.iloc[k, 6] and (k!=j) and (not np.isnan(data.iloc[k, i]))):
            for l in range(len(columns)):
                if ((k!=j) and (l!=i)):
                    if (not np.isnan(data.iloc[k,l]) and not np.isnan(data.iloc[j, i])):
                        distance += abs(data.iloc[k, l] - data.iloc[j,i])
                    elif (not np.isnan(data.iloc[j, i])):
                        distance += abs(data.iloc[j,i])
                    elif (not np.isnan(data.iloc[k, l])):
                        distance += abs(data.iloc[k,l])
            if (len(nb[0]) < 3):
                nb[0].append(k)
                nb[1].append(distance)
            else:
                for m in range(len(nb)):
                    if (distance < nb[1][m]):
                        nb[0][m] = k
                        nb[1][m] = distance
    average = (data.iloc[nb[0][0], i] + data.iloc[nb[0][1], i] + data.iloc[nb[0][2], i])/3
    return average

def fillnan():

    global data
    global data_app
    global columns
    global powers

    for i in range(len(columns)):
        for j in range(len(data[columns[i]])):
            if (np.isnan(data.iloc[j, i])):
                data.iloc[j, i] = estimate(i, j)

def test_acuracy(train_data_powered, test_data_powered):

    global data
    global data_app
    global columns
    global powers


    solves = test_data_powered.loc[:, 'Outcome']
    test_data_powered2 = test_data_powered.copy()
    test_data_powered2.pop('Outcome')
    X = train_data_powered[columns]
    y = train_data_powered.Outcome
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    y_pred = neigh.predict(test_data_powered2)
    size = len(y_pred)
    sum = 0
    for i in range(len(solves)):
        if (y_pred[i] == solves.iloc[i]):
            sum+=1
    del (test_data_powered2)
    del (solves)
    del (X)
    del (y)
    del (neigh)
    del (y_pred)
    return sum/size

def power(train_data, test_data):
    global data
    global data_app
    global columns
    global powers
    for j in range(len(columns)):
        min = train_data.iloc[:,j].min()
        if (test_data.iloc[:,j].min() < min):
            min = test_data.iloc[:,j].min()
        max = train_data.iloc[:,j].max()
        if (test_data.iloc[:,j].max() > max):
            max = test_data.iloc[:,j].max()
        scaled = []
        
        for i in range(len(train_data[columns[j]])):
            scaled.append(powers[j]*(10*((train_data.iloc[i, j]) - min)/(max - min)))
        train_data.loc[:,columns[j]] = scaled
        scaled.clear()

        for i in range(len(test_data[columns[j]])):
            scaled.append(powers[j]*(10*((test_data.iloc[i, j]) - min)/(max - min)))
        test_data.loc[:,columns[j]] = scaled
        scaled.clear()
    del (scaled)
    return (train_data,test_data)

def test_power():


    global data
    global data_app
    global columns
    global powers

    powers = [1, 1, 1, 1, 1, 1]
    best_powers = powers

    acuracy=0.0
    best_acuracy=0.0

    while (powers != [10, 10, 10, 10, 10, 10]):

        acuracy = 0.0

        for i in range(3):
            train_data, test_data = train_test_split(data, test_size=0.33, random_state=random.randint(10,100))
            train_data_powered,test_data_powered = power(train_data, test_data)
            acuracy += test_acuracy(train_data_powered, test_data_powered)
            del (train_data)
            del (test_data)
            del (train_data_powered)
            del (test_data_powered)
        acuracy/=3
        print(acuracy)
        print(powers)
        if (acuracy > best_acuracy):
            best_acuracy = acuracy
            best_powers = powers
            print('-------- NEW BEST --------')
            print('best acuracy: ' + str (acuracy))
            print('best powers: ' + str (powers))
            print('-------- NEW BEST --------')
            input('New Best')

        powers[len(powers) - 1]+=2

        for i in range((len(powers) - 1), -1, -1):
            if (powers[i] > 10):
                if (i - 1 >= 0):
                    powers[i-1]+=2
                    powers[i]=1
                else:
                    powers = [10, 10, 10, 10, 10, 10]

    powers = best_powers
    print(powers)

def repower():

    global data
    global data_app
    global columns
    global powers

    for j in range(len(columns)):
        min = data[columns[j]].min()
        if (data_app[columns[j]].min() < min):
            min = data_app[columns[j]].min()
        max = data[columns[j]].max()
        if (data_app[columns[j]].max() > max):
            max = data_app[columns[j]].max()
        scaled = []

        for i in range(len(data[columns[j]])):
            scaled.append(powers[j]*(10*((data.iloc[i, j]) - min)/(max - min)))
        data.loc[:,columns[j]] = scaled

        scaled.clear()

        for i in range(len(data_app[columns[j]])):
            scaled.append(powers[j]*(10*((data_app.iloc[i, j]) - min)/(max - min)))
        data_app.loc[:,columns[j]] = scaled

        scaled.clear()

def send():

    global data
    global data_app
    global columns
    global powers

    # Criando X and y par ao algorítmo de aprendizagem de máquina.\
    print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
    # Caso queira modificar as colunas consideradas basta algera o array a seguir.
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[feature_cols]
    y = data.Outcome


    # Ciando o modelo preditivo para a base trabalhada
    print(' - Criando modelo preditivo')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    #realizando previsões com o arquivo de
    print(' - Aplicando modelo e enviando para o servidor')
    y_pred = neigh.predict(data_app)

    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

    #TODO Substituir pela sua chave aqui
    DEV_KEY = "Flash 2.0"

    # json para ser enviado para o servidor
    data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url = URL, data = data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    print(" - Resposta do servidor:\n", r.text, "\n")

option = -1 
while(option != 0):
    print('1 - Ler os arquivos')
    print('2 - limpar as colunas')
    print('3 - preencher os dados ausentes')
    print('4 - verificar os melhores powers')
    print('5 - normalizar os valores')
    print('6 - enviar modelo')
    print('0 - SAIR')
    option = int(input('Digite:'))
    if(option == 1):
        read()

    elif(option == 2):
        clear()

    elif(option == 3):
        fillnan()

    elif(option == 4):
        test_power()

    elif(option == 5):
        repower()

    elif(option ==  6):
        send()
    