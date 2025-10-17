from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


urna=DiscreteBayesianNetwork([('Die','Extragere')])

pos=nx.circular_layout(urna)
nx.draw(urna,with_labels=True,pos=pos,alpha=0.5,node_size=2000)

#il afisam
plt.show()

#0=prim 1=sa fie 6 si 2=altcv
CPD_Die=TabularCPD(variable='Die',variable_card=3,values=[[1/2],[1/6],[1/3]])
print(CPD_Die)
#Extragere(0)=extragem bila neagra;Extragere(1)= extragem bila rosie
#Extragere(2)=extragem bila albastra
CPD_Extragere=TabularCPD(variable="Extragere",variable_card=3,
                        values=[[3/10,2/10,2/10],[3/10,4/10,3/10],[4/10,4/10,5/10]],
                        evidence=['Die'],evidence_card=[3])
print(CPD_Extragere)

urna.add_cpds(CPD_Die,CPD_Extragere)

print(urna.check_model())

#calculam inferenta, valoarea lui Extragere(1) este probabilitatea sa extragem o bila rosie
infer=VariableElimination(urna)
print(f'Probabilitatea sa extragem o bila rosie Extragere(1) \n {infer.query(variables=['Extragere'])}')

#in laboratorul trecut, probabilitatea teoretica pe care am calculat-o de a extrage o bila rosie
# era aprox 0.31, iar aici a iesit ca fiind 0.3167


