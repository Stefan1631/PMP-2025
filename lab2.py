import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

nr_R_obtinue=0
nr_repetari=0

def simulare_exp():
    dice_result = np.random.randint(1,7) #aruncam cu zarul
    print(dice_result)
    urn = ['R', 'R', 'R', 'B', 'B', 'B', 'B', 'Bl', 'Bl']  # am creat urna

    if dice_result==2 or dice_result==3 or dice_result==5: #in fnct de rezultat, adaugam bila coresp in urna
        urn.append('Bl')
    elif dice_result==6:
        urn.append('R')
    else:
        urn.append('B')

    poz=np.random.randint(0,len(urn))#selectam o bila din urna
    urn_result=urn[poz]
    print(f'am obtinut o bila {urn_result}')
    return urn_result


for i in range(1,1000):
    if simulare_exp() == 'R':
        nr_R_obtinue+=1
    nr_repetari+=1
print(f'probabilitatea de a obtine o bila rosie este {nr_R_obtinue/nr_repetari}')


# valoarea teoretica de a obtine o bila rosie:

#P(adauga bila Bl)=1/2
#P(adauga bila R)=1/6
#P(adauga bila B)=1/3

#P(extrage R | am adaugat bila Bl)=3/10
#P(extrage R | am adaugat bila R)=4/10
#P(extrage R | am adaugat bila B)=3/10

#P(extrage R)=1/2*3/10+1/6*4/10+1/3*3/10=0,31 (aprox)
