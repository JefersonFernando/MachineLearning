import requests
import random

URL = "http://localhost:8080/antenna/simulate?phi1={0}&theta1={1}&phi2={2}&theta2={3}&phi3={4}&theta3={5}"
#URL = "https://aydanomachado.com/mlclass/02_Optimization.php?phi1={0}&theta1={1}&phi2={2}&theta2={3}&phi3={4}&theta3={5}&dev_key=Flash 2.0"

class Individual():
    def __init__(self, chromosome):
        self.ranking = float(-99999)
        self.chromosome = chromosome

class settings:
    def __init__(self, phi = [], theta = [], gain = 0):
        self.phi = phi

        self.theta = theta

        self.gain = gain
    
    def getphi(self):
        return self.phi

    def setphi(self, newphi):
        self.phi = newphi
    
    def gettheta(self):
        return self.theta
    
    def settheta(self, newtheta):
        self.theta = newtheta

    def getgain(self):
        return self.gain

    def setgain(self, gain):
        self.gain = gain

def formaturl(setting):
    NURL = URL.format(setting.getphi()[0], setting.gettheta()[0], setting.getphi()[1], setting.gettheta()[1], setting.getphi()[2], setting.gettheta()[2])
    return NURL

def requestgain(setting):
    r = requests.post(url = formaturl(setting))
    gain = float(r.text.split()[0])
    print(r.text)
    return gain

#def requestgain(setting):
#    r = requests.post(url = formaturl(setting))
#    data = r.json()
#    gain = float(data["gain"])
#    print(data)
#    return gain
    
def new_setting(theta = None, phi = None, gain = None):
    new = settings()
    if(theta == None):
        theta = []
        for i in range(3):
            theta.append(random.randint(0, 360))
        new.settheta(theta)
    else:
        new.settheta(theta)

    if(phi == None):    
        phi = []
        for i in range(3):
            phi.append(random.randint(0, 360))
        new.setphi(phi)
    else:
        new.setphi(phi)

    if(gain == None):
        new.setgain(requestgain(new))
    else:
        new.setgain(gain)

    return new

def generate_settings(n):
    settings = []
    for i in range(n):
        settings.append(new_setting())
    return settings

def hill_climbing(setting):
    maxsetting = setting
    for i in range(len(setting.getphi())):
        phi = setting.getphi().copy()
        if (phi[i] < 360):
          phi[i] += 10
        
        aux = new_setting(theta = setting.gettheta().copy(),phi = phi.copy())
        if(aux.getgain() > maxsetting.getgain()):
            aux = hill_climbing(aux)
            maxsetting = aux
    
    for i in range(len(setting.gettheta())):
        theta = setting.gettheta().copy()
        if (theta[i] < 360):
          theta[i] += 10
        aux = new_setting(theta = theta.copy(), phi = setting.getphi().copy())
        if(aux.getgain() > maxsetting.getgain()):
            aux = hill_climbing(aux)
            maxsetting = aux

    for i in range(len(setting.getphi())):
        phi = setting.getphi().copy()
        if (phi[i] > 0):
          phi[i] -= 10
        aux = new_setting(theta = setting.gettheta().copy(),phi = phi.copy())
        if(aux.getgain() > maxsetting.getgain()):
            aux = hill_climbing(aux)
            maxsetting = aux
    
    for i in range(len(setting.gettheta())):
        theta = setting.gettheta().copy()
        if (theta[i] < 0):
          theta[i] -= 10
        aux = new_setting(theta = theta.copy(), phi = setting.getphi().copy())
        if(aux.getgain() > maxsetting.getgain()):
            aux = hill_climbing(aux)
            maxsetting = aux
    
    return maxsetting

def check_distance(coords, distances, setting):
    distance = 0
    for i in range(len(coords)):
        distance = 0
        for j in range(len(coords[i].getphi())):
            distance += abs(setting.getphi()[j] - coords[i].getphi()[j])
        
        for j in range(len(coords[i].gettheta())):
            distance += abs(setting.gettheta()[j] - coords[i].gettheta()[j])
        if(distance < distances[i]):
            return True
    return False

def distancemp(maximum, point):
    distance = 0
    for i in range(len(maximum.getphi())):
        distance += abs(maximum.getphi()[i] - point.getphi()[i])
    
    for i in range(len(maximum.gettheta())):
        distance += abs(maximum.gettheta()[i] - point.gettheta()[i])
    return distance


def limit_distance():
    coords = []
    distances = []
    best = []
    for i in range(10):
        print("Nova tentativa")
        new = new_setting(gain = 0.5)
        if(check_distance(coords, distances, new)):
            i-=1
            del(new)
            break
        new.setgain(requestgain(new))
        maximum = hill_climbing(new)
        distance = distancemp(maximum, new)
        coords.append(maximum)
        distances.append(distance)
        best.append(maximum)
        
    return best


def init_population(individuals, bests):
    population = []
    for i in range(0, individuals):
        chromosome = []
        for j in range(0, 3):
            theta = random.randint(0, 359)
            phi = random.randint(0, 359)
            chromosome.append((phi, theta))

        population.append(Individual(chromosome))

    for i in range(len(bests)):
        chromosome = []
        for j in range(0, 3):
            theta = bests[i].gettheta()[j]
            phi = bests[i].getphi()[j]
            chromosome.append((phi, theta))

        population.append(Individual(chromosome))
    return population


def evaluate(population):
    for individual in population:
        phi1, theta1 = individual.chromosome[0]
        phi2, theta2 = individual.chromosome[1]
        phi3, theta3 = individual.chromosome[2]
        response = requests.get(f'http://localhost:8080/antenna/simulate?phi1={phi1}&theta1={theta1}&phi2={phi2}&theta2={theta2}&phi3={phi3}&theta3={theta3}')
        #response = requests.get(f'https://aydanomachado.com/mlclass/02_Optimization.php?phi1={phi1}&theta1={theta1}&phi2={phi2}&theta2={theta2}&phi3={phi3}&theta3={theta3}&dev_key=Flash 2.0')
        format_json = json.loads(response.text)
        print(format_json)
        new_score = float(format_json['gain'])
        individual.ranking = new_score
    population.sort(key=lambda ind: ind.ranking)


def crossover(population):
    new_population = []
    for i in range(60, len(population)):
        new_population.append(population[i])

    for i in range(0, 100):
        worst = population[99-i]
        best_index = len(population)-1-i
        best = population[best_index]

        first_index = random.randint(0, 2)
        second_index = first_index
        while second_index == first_index:
            second_index = random.randint(0, 2)

        third_index = abs(first_index-second_index)
        chromosome = [best.chromosome[first_index],
                      best.chromosome[second_index], worst.chromosome[third_index]]
        new_population.append(Individual(chromosome))

    return new_population


def mutation(population):
    for i in range(0, 10):
        index = random.randint(0, len(population)-1)
        individual = population[index]
        phi1, theta1 = individual.chromosome[0]
        phi2, theta2 = individual.chromosome[1]
        phi3, theta3 = individual.chromosome[2]

        population[index] = Individual([(phi3, theta2), (phi1, theta3), (phi2, theta1)])



def genetic(bests):
    while True:
        population = init_population(80, bests)
        evaluate(population)
        cycles = 0
        while cycles != 200:
            population = crossover(population)
            mutation(population)
            evaluate(population)   
            cycles += 1
        
        individual = population[len(population)-1]
        phi1, theta1 = individual.chromosome[0]
        phi2, theta2 = individual.chromosome[1]
        phi3, theta3 = individual.chromosome[2]

        response = requests.get(f'https://aydanomachado.com/mlclass/02_Optimization.php?phi1={phi1}&theta1={theta1}&phi2={phi2}&theta2={theta2}&phi3={phi3}&theta3={theta3}&dev_key=Flash 2.0')
        print(response.text)

        

def main():
    bests = limit_distance()
    genetic(bests)




    

main()