# traffic_DQNA_Keras
A deep reinforcement learning based traffic control system


Please ensure you have Sumo v0.21 installed in your machine, you can find it here

https://sourceforge.net/projects/sumo/files/sumo/version%200.21.0/ 

and add sumo to your environment path:

https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html





To train the model , simply run traffic_control_1003


No-GUI Option:
>  traci.start(["sumo", "-c", "Intersection.sumocfg", '--start'])

GUI option:
>  traci.start(["sumo", "-c", "Intersection.sumocfg", '--start'])
  
  
To replay the agent's memory at specific generations, load the file here.
>  agent.load('Models/reinf_traf_control_194_1.h5')
