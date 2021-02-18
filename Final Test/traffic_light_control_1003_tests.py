## This is one junction code

from __future__ import absolute_import
from __future__ import print_function


import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
                 
from sumolib import checkBinary  # noqa
import traci
            

import os
import sys
import optparse
import subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from pandas import DataFrame


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

        input_3 = keras.engine.input_layer.Input(shape=(4, 1))
        x3 = keras.layers.Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(4, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')
        model.summary()
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
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
            import traci
            
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
                
        import random
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        Probs_of_lane = [1. / 10, 1. / 13, 1. / 8,1. / 18]
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
                    
        with open("Intersection.rou.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
            <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
            <route id= "LE1LE2" edges="LE1LJ1 LJ1LE2"/>
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
        lgts = []

        cellLength = 7
        offset = 11.73
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('LJ1')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('LE2LJ1')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('LE4LJ1')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('LE3LJ1')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('LE1LJ1')
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
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][ind] = 1
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
                               traci.vehicle.getLaneIndex(v)][ind] = 1
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
            light = [0,0,0,1]

        if(traci.trafficlight.getPhase('LJ1') == 4):
            light = [0,0,1,0]

        if(traci.trafficlight.getPhase('LJ1') == 6):
            light = [0,1,0,0]


                
        position = np.array(positionMatrix ,dtype=float)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocityMatrix, dtype=float)
        velocity = velocity.reshape(1, 12, 12, 1)     
        
        lgts = np.array(light)
        lgts = lgts.reshape(1, 4, 1)
#        print(lgts)
        

        return [position, velocity, lgts]
    
    def activeJunction(self):
        if(traci.trafficlight.getPhase('LJ1') == 0 or traci.trafficlight.getPhase('LJ1') == 1):
            activeJunction = ["LE1LJ1"]
            inactiveJunction = ["LE2LJ1","LE3LJ1","LE4LJ1"]
        if(traci.trafficlight.getPhase('LJ1') == 2 or traci.trafficlight.getPhase('LJ1') == 3):
            activeJunction = ["LE4LJ1"]
            inactiveJunction = ["LE1LJ1","LE2LJ1","LE3LJ1"]
        if(traci.trafficlight.getPhase('LJ1') == 4 or traci.trafficlight.getPhase('LJ1') == 5):
            activeJunction = ["LE3LJ1"]
            inactiveJunction = ["LE1LJ1","LE2LJ1","LE4LJ1"]
        if(traci.trafficlight.getPhase('LJ1') == 6 or traci.trafficlight.getPhase('LJ1') == 7):
            activeJunction = ["LE2LJ1"]
            inactiveJunction = ["LE1LJ1","LE3LJ1","LE4LJ1"]
            
        
            
        return[activeJunction,inactiveJunction]
        
    def trafficstateadd(x):
        x+=1
        
        if (x==8):
            x= 0
        return x
    
    def printval(self):
        print(sumoInt.activeJunction())
        print(traci.edge.getLastStepVehicleNumber(
                activeJunction[0]),traci.edge.getLastStepHaltingNumber(
                inactiveJunction[0]), traci.edge.getLastStepHaltingNumber(
                inactiveJunction[1]), traci.edge.getLastStepHaltingNumber(
                inactiveJunction[2]))
        print(reward1 ,reward2, action)
        
            

if __name__ == '__main__':
    sumoInt = SumoIntersection()
    

    sumoInt.generate_routefile()
    # Main logic
    # parameters
    episodes = 1
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
        log = open('testingFinal.txt', 'a')
        log1 = open ('RewardRecords.txt' , 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        stepz = 0   
        action = 0
        reward = 0
        counter = 0 
        LastAction= 0
        Totalreward =0
        waitingtime2 =0
        QueueLengthLE1=0
        QueueLengthLE2=0
        QueueLengthLE3=0
        QueueLengthLE4=0


        traci.start(["sumo-gui", "-c", "Intersection.sumocfg", '--start'])
        traci.trafficlight.setPhase("LJ1", 0)
        traci.trafficlight.setPhaseDuration("LJ1", 200)
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agent.act(state)
            light = state[2]
            activeJunction = sumoInt.activeJunction()[0]
            inactiveJunction = sumoInt.activeJunction()[1]
#            print(state)
            if (action == LastAction):
                x = traci.trafficlight.getPhase('LJ1')
                reward1 = traci.edge.getLastStepVehicleNumber(
                        activeJunction[0])
                reward2 = traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[2])

                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase('LJ1', x)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                            activeJunction[0])
                    reward2 += traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[2])
                    waiting_time+= traci.edge.getLastStepHaltingNumber(
                            activeJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[2])
                    traci.simulationStep()
                 
            if (action != LastAction):
                x = traci.trafficlight.getPhase('LJ1')
                x = SumoIntersection.trafficstateadd(x)
                activeJunction = sumoInt.activeJunction()[0]
                inactiveJunction = sumoInt.activeJunction()[1]
                
                #transition phase
                for i in range(3):
                    stepz += 1
                    traci.trafficlight.setPhase('LJ1', x)
                    waiting_time+=traci.edge.getLastStepHaltingNumber(
                            activeJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[2])
                    traci.simulationStep()
                     
                x = action*2
                traci.trafficlight.setPhase('LJ1', x)
                activeJunction = sumoInt.activeJunction()[0]
                inactiveJunction = sumoInt.activeJunction()[1]
                reward1 = traci.edge.getLastStepVehicleNumber(
                        activeJunction[0])
                reward2 = traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[2])
                #set to next phase
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase('LJ1', x)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                            activeJunction[0])
                    reward2 += traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[2])
                    waiting_time+= traci.edge.getLastStepHaltingNumber(
                            activeJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                            inactiveJunction[2])
                    traci.simulationStep()
                    
            QueueLengthLE1+=traci.edge.getLastStepVehicleNumber('LE1LJ1')
            QueueLengthLE2+=traci.edge.getLastStepVehicleNumber('LE2LJ1')
            QueueLengthLE3+=traci.edge.getLastStepVehicleNumber('LE3LJ1')
            QueueLengthLE4+=traci.edge.getLastStepVehicleNumber('LE4LJ1')
            counter += 1
            
            LastAction=action  
            new_state = sumoInt.getState()
            reward = reward1 - reward2
            Totalreward += reward

            agent.remember(state, action, reward, new_state, False)
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
#            if len(agent.memory) > batch_size:
#                agent.replay(batch_size)
        
        QueueLengthLE1 = QueueLengthLE1/counter 
        QueueLengthLE2 = QueueLengthLE2/counter
        QueueLengthLE3 = QueueLengthLE3/counter
        QueueLengthLE4 = QueueLengthLE4/counter
        print("Average Queue Length:" + '\n' + "LE1:" + str(
                QueueLengthLE1)+'\n'+ "LE2:" + str(
                QueueLengthLE2)+'\n'+ "LE3:" + str(
                QueueLengthLE3)+'\n'+ "LE4:" + str(
                QueueLengthLE4)+'\n')
        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        log.write('episode - ,' + str(e)  + ', total wa         iting time - ,' +
                  str(waiting_time) + ',reward = ,' + str(Totalreward) + '\n')
        log1.write('episode - ,' + str(e) + ',reward = ,' + str(Totalreward) + '\n')
        log.close()
        log1.close()
        print('episode - ,' + str(e) + ', total waiting time - ,' + str(waiting_time) +", Rewards," + str(Totalreward))
#        agent.save('reinf_traf_control_' + str(e) + '.h5')                    
        traci.close(wait=False)




sys.stdout.flush()


