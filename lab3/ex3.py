from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def joc():
    #dam cu banul cinstit
    coin=np.random.randint(2)
    d=0
    heads=0
    #daca am obtinut tails(0) atunci p0 da cu zarul
    if coin == 0:
        d=np.random.randint(1,7)
        for i in range(1,2*d+1):
            #p1 da cu zarul de 2*d0 ori, cu probabilitatea de a obtine heads de 4/7
            if np.random.randint(1,8) > 3:
                heads+=1
        if heads > d:
            return 1 #p1 castiga
        return 0 #p0 castiga
    else:
        d=np.random.randint(1,7)
        for i in range(1,2*d+1):
            if np.random.randint(2) == 1 :
                heads+=1
        if heads> d:
            return 0 #p0 castiga
        return 1 #p1 castiga

def simulare(repetari):
    p0=0
    p1=0
    for i in range(1,repetari+1):
        if joc() == 0 :
            p0+=1
        else :
            p1+=1
    print(f'probabilitatea ca p0 sa castige: {p0/repetari}')
    print(f'probabilitatea ca p1 sa castige: {p1/repetari}')
#a)
simulare(10000)


#b)
#C=rezultatul primei aruncari P(C=0)=P(C=1)=1/2
#H=prob sa se obtina Heads la o aruncare in a doua runda P(H=4/7|C=0)=P(H=1/2|C=1)=1 si
# P(H=4/7|C=1)=P(H=1/2|C=0)=0
#M=numarul de heads obtinut
#W= cine castiga; W=0-> p0 castiga; W=1-> p1 castiga
joc_model=DiscreteBayesianNetwork([
        ('C','H'),
        ('H','M'),
        ('D','M'),
        ('M','W'),
        ('D','W')
    ])
pos=nx.circular_layout(joc_model)
nx.draw(joc_model,with_labels=True,pos=pos,alpha=0.5,node_size=2000)
plt.show()

