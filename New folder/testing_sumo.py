from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import traci
import traci.constants as tc
import random
import numpy as np
import matplotlib.pyplot as plt

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
            a = ["LE1", "LE2","LE3","RE1","RE2","RE3"]
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
            with open("Real_Test.rou.xml", "w") as routes:
                print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
                <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
                <route id="LE1LE2" edges="LE1LJ1 LJ1LE2"/>
                <route id="LE1LE3" edges="LE1LJ1 LJ1LE3"/>
                <route id="LE1RE1" edges="LE1LJ1 LJ1RJ1 RJ1RE1"/>
                <route id="LE1RE2" edges="LE1LJ1 LJ1RJ1 RJ1RE2"/>
                <route id="LE1RE3" edges="LE1LJ1 LJ1RJ1 RJ1RE3"/>
                <route id="LE2LE1" edges="LE2LJ1 LJ1LE1"/>
                <route id="LE2LE3" edges="LE2LJ1 LJ1LE3"/>
                <route id="LE2RE1" edges="LE2LJ1 LJ1RJ1 RJ1RE1"/>
                <route id="LE2RE2" edges="LE2LJ1 LJ1RJ1 RJ1RE2"/>
                <route id="LE2RE3" edges="LE2LJ1 LJ1RJ1 RJ1RE3"/>
                <route id="LE3LE1" edges="LE3LJ1 LJ1LE1"/>
                <route id="LE3LE2" edges="LE3LJ1 LJ1LE2"/>
                <route id="LE3RE1" edges="LE3LJ1 LJ1RJ1 RJ1RE1"/>
                <route id="LE3RE2" edges="LE3LJ1 LJ1RJ1 RJ1RE2"/>
                <route id="LE3RE3" edges="LE3LJ1 LJ1RJ1 RJ1RE3"/>
                <route id="RE1LE1" edges="RE1RJ1 RJ1LJ1 LJ1LE1"/>
                <route id="RE1LE2" edges="RE1RJ1 RJ1LJ1 LJ1LE2"/>
                <route id="RE1LE3" edges="RE1RJ1 RJ1LJ1 LJ1LE3"/>
                <route id="RE1RE2" edges="RE1RJ1 RJ1RE2"/>
                <route id="RE1RE3" edges="RE1RJ1 RJ1RE3"/>
                <route id="RE2LE1" edges="RE2RJ1 RJ1LJ1 LJ1LE1"/>
                <route id="RE2LE2" edges="RE2RJ1 RJ1LJ1 LJ1LE2"/>
                <route id="RE2LE3" edges="RE2RJ1 RJ1LJ1 LJ1LE3"/>
                <route id="RE2RE1" edges="RE2RJ1 RJ1RE1"/>
                <route id="RE2RE3" edges="RE2RJ1 RJ1RE3"/>
                <route id="RE3LE1" edges="RE3RJ1 RJ1LJ1 LJ1LE1"/>
                <route id="RE3LE2" edges="RE3RJ1 RJ1LJ1 LJ1LE2"/>
                <route id="RE3LE3" edges="RE3RJ1 RJ1LJ1 LJ1LE3"/>
                <route id="RE3RE1" edges="RE3RJ1 RJ1RE1"/>
                <route id="RE3RE2" edges="RE3RJ1 RJ1RE2"/>
                ''', file=routes)
                
                lastVeh = 0
                vehNr = 0
                k=0
                i = 0
                m=0
                for i in range(N):
                    h=0
                    for m in range (6):
                        Probability_Model = Probs_of_lane[m]
                        for k in range(5):
                            if random.uniform(0, 1) <  Probability_Model:
                                print('    <vehicle id="%s_%i" type="SUMO_DEFAULT_TYPE" route="%s" depart="%i" />' % (
                                    b[h],vehNr, b[h], i), file=routes)
                                vehNr += 1
                                lastVeh = i
                            h += 1
                
                print("</routes>", file=routes)
            
            
if __name__ == '__main__':
    traci.start(["sumo-gui", "-c", "Real_Test1.sumocfg"]) 
    
    vehID = "LE1LE3_0"
    traci.vehicle.subscribe(vehID, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
    print(traci.vehicle.getSubscriptionResults(vehID))
    for step in range(500):
       print("step", step)
       traci.simulationStep()
       print(traci.vehicle.getSubscriptionResults(vehID))
    traci.close()
        
    