import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random

columns = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']

data = pd.read_csv('abalone_dataset.csv')
data_app = pd.read_csv('abalone_app.csv')

def read():
    data = pd.read_csv('abalone_dataset.csv')
    data_app = pd.read_csv('abalone_app.csv')

def convert_sex():
    data.loc[data["sex"] == 'M', 'sex'] = 1
    data.loc[data["sex"] == 'F', 'sex'] = 3
    data.loc[data["sex"] == 'I', 'sex'] = 2

    data_app.loc[data_app["sex"] == 'M', 'sex'] = 1
    data_app.loc[data_app["sex"] == 'F', 'sex'] = 3
    data_app.loc[data_app["sex"] == 'I', 'sex'] = 2   

def plot():
    plt.scatter(data.loc[data['type'] == 1, "viscera_weight"], data.loc[data['type'] == 1, "shell_weight"], c = 'r')
    plt.scatter(data.loc[data['type'] == 2, "viscera_weight"], data.loc[data['type'] == 2, "shell_weight"], c = 'b')
    plt.scatter(data.loc[data['type'] == 3, "viscera_weight"], data.loc[data['type'] == 3, "shell_weight"], c = 'g')
    plt.xlabel("viscera_weight")
    plt.ylabel("shell_weight")
    plt.show()

def normalize():
    for j in range(len(columns)):
        min = data[columns[j]].min()
        if(data_app[columns[j]].min() < min):
            min = data_app[columns[j]].min()

        max = data[columns[j]].max()
        if(data_app[columns[j]].max() > max):
            max = data_app[columns[j]].max()
        
        scaled = []

        for i in range(len(data[columns[j]])):
            scaled.append(((( data.iloc[i, j] - min) / (max - min)) * (2)) - 1)
        data.loc[:, columns[j]] = scaled.copy()

        scaled.clear()

        for i in range(len(data_app[columns[j]])):
            scaled.append(((( data_app.iloc[i, j] - min) / (max - min)) * (2)) - 1)
        data_app.loc[:, columns[j]] = scaled.copy()

        scaled.clear()

def test_acuracy():
    sum = 0
    datacopy = data.copy()

    for i in range(4):

        test_data = datacopy.iloc[i * int(len(datacopy.iloc[:, 0]) / 4) : (i + 1) * int(len(datacopy.iloc[:, 0]) / 4), :]
        train_data = pd.concat([ datacopy.iloc[0 : i * int(len(datacopy.iloc[:, 0]) / 4), :],  datacopy.iloc[(i+1) * int(len(datacopy.iloc[:, 0]) / 4) :, :]], axis = 0)

        solves = test_data.loc[:,'type']
        test_data.pop('type')
        X = train_data[columns]
        Y = train_data.type
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, Y)
        y_pred = neigh.predict(test_data)
        size = len(y_pred)

        for i in range(len(solves)):
            if (y_pred[i] == solves.iloc[i]):
                sum+=1
    
    return (sum/(len(y_pred)*4))

def check_powers():
    global data
    arquivo = open("powers.txt", "a")
    powers = [-1, -1, -1, -1, -1, -1, -0.9, -0.1]
    maximum = [ 1, 1, 1, 1, 1, 1, 1, 1]

    data_backup = data.copy()

    while(powers != maximum):
        data = data_backup.copy()
        for j in range(len(columns)):
            for i in range(len(data.iloc[:, j])):
                data.iloc[i, j] *= powers[j]

        accuracy = test_acuracy()
        arquivo.write("powers: " + str(powers) + "accuracy: " + str(accuracy)+ "\n")

        print(powers)
        print(accuracy)

        powers[len(powers) - 1] += 0.1

        for i in range((len(powers) - 1), -1, -1):
            if(powers[i] > 1):
                if(i-1 >= 0):
                    powers[i-1] += 0.1
                    powers[i] = -1
                else:
                    powers = maximum
            else: 
                break
    arquivo.close()


        

def main():
    convert_sex()
    normalize()
    check_powers()
main()


