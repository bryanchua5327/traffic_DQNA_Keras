'''
Author : Tej Patel
Contact: tej18121995@gmail.com
'''
## This is one junction code

from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import traci
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 4

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = keras.engine.input_layer.Input(shape=(12, 12, 1))
        x1 = keras.layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = keras.layers.Flatten()(x1)

        input_2 = keras.engine.input_layer.Input(shape=(12, 12, 1))
        x2 = keras.layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = keras.layers.Flatten()(x2)

        input_3 = keras.engine.input_layer.Input(shape=(2, 1))
        x3 = keras.layers.Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SumoIntersection:
    def __init__(self):
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            import os, sys
            if 'SUMO_HOME' in os.environ:
                 tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
                 sys.path.append(tools)
            else:   
                 sys.exit("please declare environment variable 'SUMO_HOME'")
                 
            from sumolib import checkBinary  # noqa
            
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
                
        import random
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        Probs_of_lane = [1. / 7, 1. / 11, 1. / 30,1. / 15,1. / 21, 1. / 27]
#        pLE1 = 1. / 7
#        pLE2 = 1. / 11
#        pLE3 = 1. / 30
#        pRE1 = 1. / 15
#        pRE2 = 1. / 21
#        pRE3 = 1. / 27
        a = ["LE1", "LE2","LE3","LE4"]
        i=0
        j=0
        counter=0
        b=[]
        for i in range(len(a)):
            for j in range(len(a)):
                if (i!=j):
#                    print ("<route id="+'''"'''+ a[i] + a[j] + ''' edges="''' + a[i] + a[j] + "/>")
                    b.append(a[i] + a[j])
                    counter = counter + 1
                    print ('''<route id= "'''+a[i]+a[j]+'''" edges="'''+a[i]+"LJ1 LJ1" + a[j]+'''"/>''')
                    
        with open("Real_Test2.rou.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
            <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
            <route id= "LE1LE3" edges="LE1LJ1 LJ1LE3"/>
            <route id= "LE1LE4" edges="LE1LJ1 LJ1LE4"/>
            <route id= "LE2LE1" edges="LE2LJ1 LJ1LE1"/>
            <route id= "LE2LE3" edges="LE2LJ1 LJ1LE3"/>
            <route id= "LE2LE4" edges="LE2LJ1 LJ1LE4"/>
            <route id= "LE3LE1" edges="LE3LJ1 LJ1LE1"/>
            <route id= "LE3LE2" edges="LE3LJ1 LJ1LE2"/>
            <route id= "LE3LE4" edges="LE3LJ1 LJ1LE4"/>
            <route id= "LE4LE1" edges="LE4LJ1 LJ1LE1"/>
            <route id= "LE4LE2" edges="LE4LJ1 LJ1LE2"/>
            <route id= "LE4LE3" edges="LE4LJ1 LJ1LE3"/>
            ''', file=routes)
            
            lastVeh = 0
            vehNr = 0
            k=0
            i = 0
            m=0
            for i in range(N):
                h=0
                for m in range (4):
                    Probability_Model = Probs_of_lane[m]
                    for k in range(3):
                        if random.uniform(0, 1) <  Probability_Model:
                            print('    <vehicle id="%s_%i" type="SUMO_DEFAULT_TYPE" route="%s" depart="%i" />' % (
                                b[h],vehNr, b[h], i), file=routes)
                            vehNr += 1
                            lastVeh = i
                        h += 1
            
            print("</routes>", file=routes)
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('LJ1')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('LE1LJ1')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('LE2LJ1')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('LE3LJ1')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('LE4LJ1')
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('LJ1')[1]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road4:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = []
        if(traci.trafficlight.getPhase('LJ1') == 0):
            light = [1,0,0,0]
        if(traci.trafficlight.getPhase('LJ1') == 2):
            light = [0,1,0,0]
        if(traci.trafficlight.getPhase('LJ1') == 4):
            light = [0,0,1,0]
        if(traci.trafficlight.getPhase('LJ1') == 6):
            light = [0,0,0,1]
                
        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)
        
        

        lgts = np.array(light)
        lgts = lgts.reshape(1, 4, 1)
        

        return [position, velocity, lgts, positionMatrix,velocityMatrix]
    
    def visualize_eyes(self,Vel):

        self.Vel=Vel
        plt.matshow(Vel, cmap=plt.cm.Blues)
        plt.show
        

if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    # Main logic
    # parameters
    episodes = 2000
    batch_size = 32

    tg = 10
    ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/reinf_traf_control.h5')
    except:
        print('No models found')

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        #log = open('log.txt', 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        total_reward = reward1 - reward2
        stepz = 0
        action = 0


        traci.start([sumoCmd, "-c", "Real_Test1.sumocfg", '--start'])
        traci.trafficlight.setPhase("LJ1", 0)
        traci.trafficlight.setPhaseDuration("LJ1", 200)
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 7000:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agent.act(state)
            light = state[2]
#            two = sumoInt.visualize_eyes(Vel=velocityMatrix)

        traci.close(wait=False)

sys.stdout.flush()
