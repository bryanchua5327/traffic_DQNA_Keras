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

import h5py
from collections import deque


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

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    
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
    

    episodes = 1
    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        log = open('TLCCONSTANT.txt', 'a')
        step = 0
        waiting_time = 0
        stepz = 0   
        action = 0
        counter=0
        travel_timeLE1=0
        travel_timeLE2=0
        travel_timeLE3=0
        travel_timeLE4=0
        waiting_timeLE1=0
        waiting_timeLE2=0
        waiting_timeLE3=0
        waiting_timeLE4=0
        QueueLengthLE1=0
        QueueLengthLE2=0
        QueueLengthLE3=0
        QueueLengthLE4=0

        traci.start(["sumo-gui", "-c", "Intersection.sumocfg", '--start'])
        traci.trafficlight.setPhase("LJ1", 0)
        traci.trafficlight.setPhaseDuration("LJ1", 200)
        while traci.simulation.getMinExpectedNumber() > 0 :
            activeJunction = sumoInt.activeJunction()[0]
            inactiveJunction = sumoInt.activeJunction()[1]
            waiting_time+= traci.edge.getLastStepHaltingNumber(
                        activeJunction[0])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[0])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[1])+traci.edge.getLastStepHaltingNumber(
                        inactiveJunction[2])
            
            waiting_timeLE1=+traci.edge.getWaitingTime('LE1LJ1')
            waiting_timeLE2=+traci.edge.getWaitingTime('LE2LJ1')
            waiting_timeLE3=+traci.edge.getWaitingTime('LE3LJ1')
            waiting_timeLE4=+traci.edge.getWaitingTime('LE4LJ1')
            travel_timeLE1=traci.lane.getTraveltime(
                    'LE1LJ1_0')+ traci.lane.getTraveltime(
                    'LE1LJ1_1')+ traci.lane.getTraveltime(
                    'LE1LJ1_2')
            travel_timeLE2=traci.lane.getTraveltime(
                    'LE2LJ1_0')+ traci.lane.getTraveltime(
                    'LE2LJ1_1')+ traci.lane.getTraveltime(
                    'LE2LJ1_2')
            travel_timeLE3=traci.lane.getTraveltime(
                    'LE3LJ1_0')+ traci.lane.getTraveltime(
                    'LE3LJ1_1')+ traci.lane.getTraveltime(
                    'LE3LJ1_2')
            travel_timeLE4=traci.lane.getTraveltime(
                    'LE4LJ1_0')+ traci.lane.getTraveltime(
                    'LE4LJ1_1')+ traci.lane.getTraveltime(
                    'LE4LJ1_2')
            QueueLengthLE1+=traci.edge.getLastStepVehicleNumber('LE1LJ1')
            QueueLengthLE2+=traci.edge.getLastStepVehicleNumber('LE2LJ1')
            QueueLengthLE3+=traci.edge.getLastStepVehicleNumber('LE3LJ1')
            QueueLengthLE4+=traci.edge.getLastStepVehicleNumber('LE4LJ1')
            
            
            counter+=1         
            
#            print("LE1 :," + str(waiting_timeLE1) +',' + str(travel_timeLE1) +',' + str(
#                    QueueLengthLE1)+'\n'+ ",LE2 :," + str(waiting_timeLE2) +',' + str(travel_timeLE2)+',' + str(
#                    QueueLengthLE2)+'\n'+ ",LE3 :," + str(waiting_timeLE3) +',' + str(travel_timeLE3)+',' + str(
#                    QueueLengthLE3)+'\n'+ ",LE4 :," + str(waiting_timeLE4) +',' + str(travel_timeLE4)+',' + str(
#                    QueueLengthLE4)+'\n')
#            log.write("LE1 :," + str(waiting_timeLE1) +',' + str(travel_timeLE1) +',' + str(
#                    QueueLengthLE1)+ ",LE2 :," + str(waiting_timeLE2) +',' + str(travel_timeLE2)+',' + str(
#                    QueueLengthLE2)+ ",LE3 :," + str(waiting_timeLE3) +',' + str(travel_timeLE3)+',' + str(
#                    QueueLengthLE3)+ ",LE4 :," + str(waiting_timeLE4) +',' + str(travel_timeLE4)+',' + str(
#                    QueueLengthLE4)+'\n')
            
            
            
            
            stepz += 1

            traci.simulationStep()
                    
                
        QueueLengthLE1 = QueueLengthLE1/counter 
        QueueLengthLE2 = QueueLengthLE2/counter
        QueueLengthLE3 = QueueLengthLE3/counter
        QueueLengthLE4 = QueueLengthLE4/counter     
        print("Average Queue Length:" + '\n' + "LE1:" + str(
                QueueLengthLE1)+'\n'+ "LE2:" + str(
                QueueLengthLE2)+'\n'+ "LE3:" + str(
                QueueLengthLE3)+'\n'+ "LE4:" + str(
                QueueLengthLE4)+'\n')
        print('episode - ,' + str(e)  + ', total waiting time - ,' +
                  str(waiting_time))
        log.write('episode - ,' + str(e)  + ', total waiting time - ,' +
                  str(waiting_time))

        log.close()
            
        traci.close(wait=False)
sys.stdout.flush()

