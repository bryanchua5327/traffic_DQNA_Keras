# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:29:03 2019

@author: user-pc
"""
trafficphase = 0
stepz=1
waiting_time=0
if (action == 1):
    trafficphase = trafficphase + 1
    for i in range(3):
        stepz += 1
        waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
            'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
        traci.trafficlight.setPhase('LJI',trafficphase)
        traci.simulationStep()
    trafficphase = trafficphase + 1
    traci.traffilight.setPhase('LJ1',trafficphase)

if (action == 0):
    for i in range(5):
        stepz += 1
        waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
            'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
        traci.trafficlight.setPhase('LJI',trafficphase)
        traci.simulationStep()
        
        
        
        
#            if(action == 0 and light[0][0][0] == 1):
##                # Transition Phase
#                for i in range(6):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 0)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
#                        'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
#                    traci.simulationStep()
#            if(action == 1 and light[0][0][0] == 1):
#                for i in range(3):
#                    traci.trafficlight.setPhase('LJ1', 1)
#                traci.trafficlight.setPhase('LJ1',2)
#            if(action == 1 and light[0][1][0] == 1):
#                for i in range(6):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 1)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
#                        'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
#                    traci.simulationStep()
                
#                # Transition Phase          
#                for i in range(10):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 2)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
#                        'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
#                    traci.simulationStep()
#                for i in range(6):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 3)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('LE1LJ1') + traci.edge.getLastStepHaltingNumber(
#                        'LE2LJ1') + traci.edge.getLastStepHaltingNumber('LE3LJ1') + traci.edge.getLastStepHaltingNumber('LE4LJ1'))
#                    traci.simulationStep()

#                # Action Execution
#                reward1 = traci.edge.getLastStepVehicleNumber(
#                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
#                reward2 = traci.edge.getLastStepHaltingNumber(
#                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
#                for i in range(10):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 4)
#                    reward1 += traci.edge.getLastStepVehicleNumber(
#                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
#                    reward2 += traci.edge.getLastStepHaltingNumber(
#                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#
#            if(action == 0 and light[0][0][0] == 1):
#                # Action Execution, no state change
#                reward1 = traci.edge.getLastStepVehicleNumber(
#                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
#                reward2 = traci.edge.getLastStepHaltingNumber(
#                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
#                for i in range(10):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 4)
#                    reward1 += traci.edge.getLastStepVehicleNumber(
#                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
#                    reward2 += traci.edge.getLastStepHaltingNumber(
#                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#
#            if(action == 1 and light[0][0][0] == 0):
#                # Action Execution, no state change
#                reward1 = traci.edge.getLastStepVehicleNumber(
#                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
#                reward2 = traci.edge.getLastStepHaltingNumber(
#                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
#                for i in range(10):
#                    stepz += 1
#                    reward1 += traci.edge.getLastStepVehicleNumber(
#                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
#                    reward2 += traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
#                    traci.trafficlight.setPhase('LJ1', 0)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#
#            if(action == 1 and light[0][0][0] == 1):
#                for i in range(6):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 5)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#                for i in range(10):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 6)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#                for i in range(6):
#                    stepz += 1
#                    traci.trafficlight.setPhase('LJ1', 7)
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#
#                reward1 = traci.edge.getLastStepVehicleNumber(
#                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
#                reward2 = traci.edge.getLastStepHaltingNumber(
#                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
#                for i in range(10):
#                    stepz += 1
#                    traci.trafficlight.setPhase('0', 0)
#                    reward1 += traci.edge.getLastStepVehicleNumber(
#                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
#                    reward2 += traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
#                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
#                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
#                    traci.simulationStep()
#   

#
