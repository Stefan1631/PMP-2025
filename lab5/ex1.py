import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import networkx as nx

#subpunctul a)

#starile ascunse
#easy=0,medium=1,hard=2
states=['Easy','Medium','Hard']
n_states=len(states)

#observatiile
#FB=0,B=1,S=2,NS=3
observations=['FB','B','S','NS']
n_observations=len(observations)

#probabilitatile initiale pt starile ascunse
initial_state_probability=np.array([1/3,1/3,1/3])

#probabilitatea de a trece din starea i in starea j
transition_probability=np.array([
    [1/4,1/4,1/2],
    [1/4,1/4,1/2],
    [1/2,1/2,0]
 ]
)

#probabilitatea sa observam j in starea i
emission_probability=np.array([
    [0.2,0.3,0.4,0.1],
    [0.15,0.25,0.5,0.1],
    [0.1,0.2,0.4,0.3]
])


#definim modelul
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = initial_state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#facem graful
G=nx.DiGraph()
#adaugam starile in graf
for state in states:
    G.add_node(state)

#adaugam starile observate in graf
for state in observations:
    G.add_node(state)

#adaugam arcele bazate pe matricea de tranzitie
for i in range(n_states):
    for j in range(n_states):
        if transition_probability[i][j]>0:
            G.add_edge(states[i],states[j],weight=transition_probability[i][j])

#adaugam arcele intre starile ascunse si observatiile corespunzatoare
for i in range(n_states):
    for j in range(n_observations):
        if emission_probability[i][j]>0:
            G.add_edge(states[i],observations[j],weight=emission_probability[i][j])

#afisam graful
pos = nx.shell_layout(G, nlist=[states, observations])
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos,nodelist=states,node_size=1000, node_color="skyblue", alpha=0.9)
nx.draw_networkx_nodes(G, pos,nodelist=observations,node_size=1000, node_color="yellow", alpha=0.9)

edge_labels = {
    (u, v): f"{d['weight']:.2f}"
    for u, v, d in G.edges(data=True)
}

nx.draw_networkx_edges(G, pos, arrowsize=20, edge_color="gray", width=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,font_color='red', label_pos=0.3)
plt.show()

#subpunctul b)
observation_sequence=np.array([0,0,2,1,1,2,1,1,3,1,1]).reshape(-1,1)
prob_obs = model.score(observation_sequence, lengths = len(observation_sequence))
print('Probabilitatea de a obtine secventa de observatii data este:',np.exp(prob_obs))

#subpunctul c)
log_probability, hidden_states = model.decode(observation_sequence,
                                              lengths = len(observation_sequence),
                                              algorithm ='viterbi')
print("Cele mai probabile stari ascunse conform starilor observate:\n", hidden_states)
print('Probabilitatea acestor stari ascunse este:\n', np.exp(log_probability))
