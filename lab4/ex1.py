import math

from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.inference import VariableElimination

#subpunctul a)

#definim modelul markov
model = MarkovNetwork([('A1', 'A2'),
                       ('A1', 'A3'),
                       ('A2','A4'),
                       ('A2','A5'),
                       ('A3','A4'),
                       ('A4','A5')])


#afisam reteaua
pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()


#afisam clicile
clici=list(nx.find_cliques(model))
print(f"clicile sunt:\n{clici}")

#subpunctul b)

#calculam valorile functiei phi pt fiecare clica
phi_A1_A2=[math.exp(1*(-1)+2*(-1)),
           math.exp(1*(-1)+2*(1)),
           math.exp(1*1+2*(-1)),
           math.exp(1*1+2*1)]

phi_A1_A3=[math.exp(1*(-1)+3*(-1)),
           math.exp(1*(-1)+3*1),
           math.exp(1*1+3*(-1)),
           math.exp(1*1+3*1)]

phi_A4_A3=[math.exp(4*(-1)+3*(-1)),
           math.exp(4*(-1)+3*1),
           math.exp(4*1+3*(-1)),
           math.exp(4*1+3*1)]

phi_A2_A4_A5=[math.exp(2*(-1)+4*(-1)+5*(-1)),
              math.exp(2*(-1)+4*(-1)+5),
              math.exp(2*(-1)+4+5*(-1)),
              math.exp(2*(-1)+4+5),
              math.exp(2+4*(-1)+5*(-1)),
              math.exp(2+4*(-1)+5),
              math.exp(2+4+5*(-1)),
              math.exp(2+4+5)]


factor_A1_A2=DiscreteFactor(variables=['A1','A2'],cardinality=[2,2],values=phi_A1_A2)
factor_A1_A3=DiscreteFactor(variables=['A1','A3'],cardinality=[2,2],values=phi_A1_A3)
factor_A4_A3=DiscreteFactor(variables=['A4','A3'],cardinality=[2,2],values=phi_A4_A3)
factor_A2_A5_A4=DiscreteFactor(variables=['A2','A4','A5'],cardinality=[2,2,2],values=phi_A2_A4_A5)

model.add_factors(factor_A1_A2, factor_A1_A3,factor_A4_A3, factor_A2_A5_A4)
print(model.get_factors())

bp_infer = BeliefPropagation(model)
marginals = bp_infer.map_query(variables=['A1','A2','A3','A4','A5'])
print(f"config cea mai buna: {marginals}")


#afisam inferenta
infer=VariableElimination(model)
print(infer.query(variables=['A1','A2','A3','A4','A5']))
