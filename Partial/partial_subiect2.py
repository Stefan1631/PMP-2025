import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import networkx as nx


states=['W','R','S']
n_states=len(states)

observations=['L','M','H']
n_observations=len(observations)

#prob initiale pt starile ascunse
initial_state_prob=np.array([0.4,0.3,0.3])

#probabilitatea de a trece din starea i in starea j
#W=0, R=1, S=2
transition_probability=np.array([
    [0.6,0.3,0.1],
    [0.2,0.7,0.1],
    [0.3,0.2,0.5]
])

#prob sa observan j in starea i
# L=0, M=1, H=2
emission_probability=np.array(
    [
        [0.1,0.7,0.2],
        [0.05,0.25,0.7],
        [0.8,0.15,0.05]
    ]
)


#definim modelul
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = initial_state_prob
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#prob sa observam Medium,High,Low
observation_sequence=np.array([1,2,0]).reshape(-1,1)
prob_obs=np.exp(model.score(observation_sequence,lengths=len(observation_sequence)))
print(prob_obs)

#prob celor mai posibile stari ascunse bazate pe obsv de mai sus folosind alg Verterbi
log_probability, hidden_states = model.decode(observation_sequence,
                                              lengths = len(observation_sequence),
                                              algorithm ='viterbi')
print(hidden_states)

#Viterbi e preferat over Brute force deoarece pt multe var aleatoare, brute force trb sa calculeze
#nenumarate prob conditionale, tabelele respective devenind f. mari


