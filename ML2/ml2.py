import requests
import random
import json

URL = "http://localhost:8080/antenna/simulate?phi1={0}&theta1={1}&phi2={2}&theta2={3}&phi3={4}&theta3={5}"

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
    return gain
    
def new_setting():
    new = settings()

    theta = []
    for i in range(3):
        theta.append(random.randint(0, 360))
    new.settheta(theta)

    phi = []
    for i in range(3):
        phi.append(random.randint(0, 360))
    new.setphi(phi)

    new.setgain(requestgain(new))
    return new

def generate_settings(n):
    settings = []
    for i in range(n):
        settings.append(new_setting())
    return settings

def sort_settings(settings):
    aux = settings.copy()
    for i in range (len(aux)):
        gain = int(aux[i].getgain())
        for j in range(gain):
            aux.append(aux[i].copy())

    

def genetic():
    settings = generate_settings(100)
    sorted = sort_settings(settings)

def main():
    genetic()
main()