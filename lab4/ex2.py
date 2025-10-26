import math

from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.inference import VariableElimination
import numpy as np

#subpunctul a)
n=5
model = MarkovNetwork()
vecini=[]
lamda=4

#adaugam in matricea de vecini pixelii
for i in range(1,n+1):
    rand=[]
    for j in range(1,n+1):
        rand.append(f'x{(i-1)*n+j}')
    vecini.append(rand)

#stabilim vecinii in model dintre pixeli
for i in range(n):
    for j in range(n):
        if i > 0:
            model.add_edge(vecini[i][j], vecini[i - 1][j])
        if i < n-1:
            model.add_edge(vecini[i][j], vecini[i + 1][j])
        if j > 0:
            model.add_edge(vecini[i][j], vecini[i][j - 1])
        if j < n-1:
            model.add_edge(vecini[i][j], vecini[i][j + 1])


original = np.random.randint(0, 2, size=(n, n))
noisy = original.copy()

#introducem zgomotul
#alegem random 10% din noduri
nodes = np.random.choice(n, int(0.1 * n*n), replace=False)
for node in nodes:
    i=node//n
    j=node%n
    noisy[i][j] = 1 - noisy[i][j]

print('Imaginea originala:\n', original)
print('Imaginea zgomotoasa:\n', noisy)


#cream factorii
factori=[]
#P(x)=1/z*exp(-E(x))=1/z*exp(- prima suma)*exp(- a doua suma)
#prima suma:
for i in range(n):
    for j in range(n):
        values=[]
        for x in [0,1]:
            values.append(math.exp(-lamda*(x-noisy[i][j])**2))
        factori.append(DiscreteFactor([vecini[i][j]],[2],values))

#a doua suma:
for (i,j) in model.edges():
    values = []
    for xi in [0, 1]:
        for xj in[0,1]:
            values.append(math.exp(-(xi-xj) ** 2))
    factori.append(DiscreteFactor([i,j], [2,2], values))

#adaugam factorii
model.add_factors(*factori)
#afisam reteaua
pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

print(model.check_model())




