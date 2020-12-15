import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random

#powers = [-0.001959296817599432, 0.40178398088632017, 0.720954829979833, 0.29619846710817765, 0.5467256132143437, 0.5097512318998636, -0.027161611960752063, 0.5594433590342884]   #Accuracy:0.6481481481481481

columns = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']

data = pd.read_csv('abalone_dataset.csv')
data_app = pd.read_csv('abalone_app.csv')

class individual():
    def __init__(self, chromosome, k):
        self.ranking = float(-99999)
        self.chromosome = chromosome
        self.k = k

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

def potentialize():
    for i in range(len(columns)):
        for j in range(len(data.iloc[:, i])):
            data.iloc[j, i] *= powers[i]
    
    for i in range(len(columns)):
        for j in range(len(data_app.iloc[:, i])):
            data_app.iloc[j, i] *= powers[i]

def test_acuracy(k):
    sum = 0
    datacopy = data.copy()

    for i in range(4):

        test_data = datacopy.iloc[i * int(len(datacopy.iloc[:, 0]) / 4) : (i + 1) * int(len(datacopy.iloc[:, 0]) / 4), :]
        train_data = pd.concat([ datacopy.iloc[0 : i * int(len(datacopy.iloc[:, 0]) / 4), :],  datacopy.iloc[(i+1) * int(len(datacopy.iloc[:, 0]) / 4) :, :]], axis = 0)

        solves = test_data.loc[:,'type']
        test_data.pop('type')
        X = train_data[columns]
        Y = train_data.type
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, Y)
        y_pred = neigh.predict(test_data)
        size = len(y_pred)

        for i in range(len(solves)):
            if (y_pred[i] == solves.iloc[i]):
                sum+=1
    
    return (sum/(len(y_pred)*4))

def submit():
    global data
    global data_app
    X = data[columns]
    Y = data.type
    neigh = KNeighborsClassifier(n_neighbors=15)
    neigh.fit(X, Y)
    y_pred = neigh.predict(data_app)
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"

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

def init_population(individuals):
    population = []

    for i in range(individuals):
        chromosome = []
        for j in range(len(columns)):
            chromosome.append(random.uniform(-1.0, 1.0))
        k = random.randint(3, 30)
        population.append(individual(chromosome, k))
    
    return population

def evaluate(population):
    global data
    arquivo = open("powers2.txt", "a")
    for individual in population:
        if(individual.ranking < 0):
            datacopy = data.copy()
            powers = individual.chromosome
            for i in range(len(columns)):
                for j in range(len(data.iloc[:, i])):
                    data.iloc[j, i] *= powers[i]
            individual.ranking = test_acuracy(individual.k)
            del(data)
            data = datacopy
            arquivo.write("powers:" + str(powers) + "K:" + str(individual.k) + "   Accuracy:" + str(individual.ranking) + "\n")
            print("powers:" + str(powers) + "K:" + str(individual.k) + "    Accuracy:" + str(individual.ranking) + "\n")
    arquivo.close
    population.sort(key = lambda ind: ind.ranking)
    
def crossover(population):
    new_population = []
    chromosome=[]
    index = []
    for i in range(int(len(population)/2), len(population)):
        new_population.append(population[i])

    for i in range(int(len(population)/2)):
        index.append(random.randint(0, int(len(population)/2)))
        index.append(random.randint(int(len(population)/2) + 1, len(population) - 1))
        del(chromosome)
        chromosome = []
        for j in range(len(columns)):
            ind = population[index[random.randint(0,1)]]
            chromosome.append(ind.chromosome[j])
        k = ind.k
        
        new_population.append(individual(chromosome, k))
    del(population)
    return new_population

def mutation(population):
    for i in range(10):
        individual = population[random.randint(0,len(population) - 3)]
        chromosome = individual.chromosome
        for j in range(len(chromosome)):
            chromosome[j] *=  random.uniform(0.8, 1.2)
        individual.chromosome = chromosome


def genetic():
    population = init_population(100)
    for k in range(200):
        print("----- NOVA GERAÇÃO -----")
        evaluate(population)
        arquivo = open("powers2.txt", "a")
        arquivo.write("BEST:  " + str(population[-1].chromosome) + "K:" + str(population[-1].k) + "    Accuracy:" + str(population[-1].ranking) + "\n")
        arquivo.close()
        print("BEST:  " + str(population[-1].chromosome) + "K:" + str(population[-1].k) + "    Accuracy:" + str(population[-1].ranking) + "\n")
        population = crossover(population)
        mutation(population)

def main():
    convert_sex()
    normalize()
    #potentialize()
    #submit()
    genetic()

main()