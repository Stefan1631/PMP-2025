import numpy as np
import pymc as pm
import arviz as az
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


#definim modelul
model=DiscreteBayesianNetwork([
    ('O','H'),
    ('O','W'),
    ('H','R'),
    ('W','R'),
    ('H','E'),
    ('R','C')
])
pos=nx.circular_layout(model)
nx.draw(model,with_labels=True,pos=pos,alpha=0.5,node_size=2000)

#il afisam
plt.show()



#valorile de 0 corespund primei valori pt fiecare variabila discreta, iar valorile de 1 corespund
#celei de a 2a valoare pt fiecare var discreta

#radacina O
CPD_O=TabularCPD(variable='O',variable_card=2,values=[[0.3],[0.7]])

#noduri cu un parinte
CPD_H=TabularCPD(variable='H',variable_card=2,values=[[0.9,0.2],[0.1,0.8]],
                 evidence=['O'],evidence_card=[2])

CPD_W=TabularCPD(variable='W',variable_card=2,values=[[0.1,0.6],[0.9,0.4]],
                 evidence=['O'],evidence_card=[2])
#cu doi parinti :R
CPD_R=TabularCPD(variable='R',variable_card=2,
                 values=[[0.6,0.9,0.3,0.5],[0.4,0.1,0.7,0.5]],
                 evidence=['H','W'],evidence_card=[2,2])

#iar cu un parinte
CPD_E=TabularCPD(variable='E',variable_card=2,values=[[0.8,0.2],[0.2,0.8]],
                 evidence=['H'],evidence_card=[2])

CPD_C=TabularCPD(variable='C',variable_card=2,values=[[0.85,0.4],[0.15,0.6]],
                 evidence=['R'],evidence_card=[2])

model.add_cpds(CPD_O,CPD_W,CPD_H,CPD_R,CPD_E,CPD_C)

print(CPD_O)
print(CPD_H)
print(CPD_W)
print(CPD_R)
print(CPD_E)
print(CPD_C)

print(model.check_model())


#facem inferenta folosind VariableElimination

infer=VariableElimination(model)

probH=infer.query(variables=['H'],evidence={'C':0})
probE=infer.query(variables=['E'],evidence={'C':0})
probHW=infer.max_marginal(variables=['H','W'],evidence={'C':0})

#valorile cautate sunt E(0) si H(0)
print(probH)
print(probE)

print(probHW)


#print(model.local_independencies(['O','H','W','R','C']))


#W este independent conditional de E stiind H pt ca W nu este nici efect indirect a lui E,nu
#nici cauza a lui E si nu sunt nici cauze comune ale lui H.

#O si C nu sunt independente conditional cand e obsv R deoarece C este o cauza a lui O,
#iar R blocheaza drumul