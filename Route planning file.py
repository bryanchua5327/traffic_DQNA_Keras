# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:06:04 2019

@author: user
"""
one=0
if(one==0):
        import numpy as np
        np.random.seed(42)
        No_of_Cars = [1300,1000,1500,900] #in an hour LE1-LE4
        p1 = np.random.poisson(No_of_Cars[0]/3600, 3600)
        p2 = np.random.poisson(No_of_Cars[1]/3600, 3600)
        p3 = np.random.poisson(No_of_Cars[2]/3600, 3600)
        p4 = np.random.poisson(No_of_Cars[3]/3600, 3600)
        Prob_of_Lane=[p1,p2,p3,p4]
#       Probs_of_lane = [1. / 10, 1. / 13, 1. / 8,1. / 18]        
        one=0
        if(one==0):
                import random
                from random import randint
                random.seed(42)  # make tests reproducible

        
        
                a = ["LE1", "LE2","LE3","LE4"]
                i=0
                j=0
                counter=0
                b=[]
                for i in range(len(a)):  
                    for j in range(len(a)):
                        if (i!=j):
                            b.append(a[i] + a[j])
                            counter = counter + 1
                            print ('''<route id= "'''+a[i]+a[j]+'''" edges="'''+a[i]+"LJ1 LJ1" + a[j]+'''"/>''')
                
                with open("Intersection.rou.xml", "w") as routes:
                    print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
                    <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
                    <route id= "LE1LE2" edges="LE1LJ1 LJ1LE2" departLane="0"/>
                    <route id= "LE1LE3" edges="LE1LJ1 LJ1LE3" departLane="1"/>
                    <route id= "LE1LE4" edges="LE1LJ1 LJ1LE4" departLane="2"/>
                    <route id= "LE2LE1" edges="LE2LJ1 LJ1LE1" departLane="2"/>
                    <route id= "LE2LE3" edges="LE2LJ1 LJ1LE3" departLane="0"/>
                    <route id= "LE2LE4" edges="LE2LJ1 LJ1LE4" departLane="1"/>
                    <route id= "LE3LE1" edges="LE3LJ1 LJ1LE1" departLane="1"/>
                    <route id= "LE3LE2" edges="LE3LJ1 LJ1LE2" departLane="2"/>
                    <route id= "LE3LE4" edges="LE3LJ1 LJ1LE4" departLane="0"/>
                    <route id= "LE4LE1" edges="LE4LJ1 LJ1LE1" departLane="0"/>
                    <route id= "LE4LE2" edges="LE4LJ1 LJ1LE2" departLane="1"/>
                    <route id= "LE4LE3" edges="LE4LJ1 LJ1LE3" departLane="2"/>
                  ''', file=routes)
            
                    calculator=[0,0,0,0]
                    Counter=[0,0,0,0,0,0,0,0,0,0,0,0]
                    h=0
                    vehNr=0
                    for k in range(3600):
                        for m in range(4):
                            counter=0
                            for n in range(Prob_of_Lane[m][k]):
                                h = randint(0,2)+(m*3)
                                print('    <vehicle id="%s_%i_%i" type="SUMO_DEFAULT_TYPE" route="%s" depart="%i" />' % (
                                        b[h],vehNr,counter, b[h], k), file=routes)
                                calculator[m] += 1
                                Counter[h] += 1
                                vehNr += 1
                                counter +=1
                
                
                    print("</routes>", file=routes)
        #            print(Counter)
        #            Total_LE1= Counter[0]+Counter[1]+Counter[2]
        #            Total_LE2= Counter[3]+Counter[4]+Counter[5]
        #            Total_LE3= Counter[6]+Counter[7]+Counter[8]
        #            Total_LE4= Counter[9]+Counter[10]+Counter[11]
        #            print(sum(Counter) , Total_LE1,Total_LE2,Total_LE3,Total_LE4)