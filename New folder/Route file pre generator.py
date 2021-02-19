# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 19:23:31 2019

@author: user-pc
"""

a = ["LE1", "LE2","LE3","RE1","RE2","RE3"]

#with open("something.txt", "w"):
i=0
j=0
counter=0
for i in range(len(a)):
    for j in range(len(a)):
        if (i!=j):
#            print ("<route id="+'''"'''+ a[i] + a[j] + ''' edges="''' + a[i] + a[j] + "/>")
            b= [a[i] + a[j]]
            print (b)
            
            counter = counter + 1

    
    
print(counter)

<route id="LE1LE2 edges="LE1LJ1 LJ1LE2/>
<route id="LE1LE3 edges="LE1LJ1 LJ1LE3/>
<route id="LE1RE1 edges="LE1LJ1 LJ1RJ1 RJ1RE1/>
<route id="LE1RE2 edges="LE1LJ1 LJ1RJ1 RJ1RE2 />
<route id="LE1RE3 edges="LE1LJ1 LJ1RJ1 RJ1RE3/>
<route id="LE2LE1 edges="LE2LJ1 LJ1LE1/>
<route id="LE2LE3 edges="LE2LJ1 LJ1LE3/>
<route id="LE2RE1 edges="LE2LJ1 LJ1RJ1 RJ1RE1/>
<route id="LE2RE2 edges="LE2LJ1 LJ1RJ1 RJ1RE2/>
<route id="LE2RE3 edges="LE2LJ1 LJ1RJ1 RJ1RE3/>
<route id="LE3LE1 edges="LE3LJ1 LJ1LE1/>
<route id="LE3LE2 edges="LE3LJ1 LJ1LE2/>
<route id="LE3RE1 edges="LE3LJ1 LJ1RJ1 RJ1RE1/>
<route id="LE3RE2 edges="LE3LJ1 LJ1RJ1 RJ1RE2/>
<route id="LE3RE3 edges="LE3LJ1 LJ1RJ1 RJ1RE3/>
<route id="RE1LE1 edges="RE1RJ1 RJ1LJ1 LJ1LE1/>
<route id="RE1LE2 edges="RE1RJ1 RJ1LJ1 LJ1LE2/>
<route id="RE1LE3 edges="RE1RJ1 RJ1LJ1 LJ1LE3/>
<route id="RE1RE2 edges="RE1RJ1 RJ1RE2/>
<route id="RE1RE3 edges="RE1RJ1 RJ1RE3/>
<route id="RE2LE1 edges="RE2RJ1 RJ1LJ1 LJ1LE1/>
<route id="RE2LE2 edges="RE2RJ1 RJ1LJ1 LJ1LE2/>
<route id="RE2LE3 edges="RE2RJ1 RJ1LJ1 LJ1LE3/>
<route id="RE2RE1 edges="RE2RJ1 RJ1RE1/>
<route id="RE2RE3 edges="RE2RJ1 RJ1RE3/>
<route id="RE3LE1 edges="RE3RJ1 RJ1LJ1 LJ1LE1/>
<route id="RE3LE2 edges="RE3RJ1 RJ1LJ1 LJ1LE2/>
<route id="RE3LE3 edges="RE3RJ1 RJ1LJ1 LJ1LE3/>
<route id="RE3RE1 edges="RE3RJ1 RJ1RE1/>
<route id="RE3RE2 edges="RE3RJ1 RJ1RE2/>