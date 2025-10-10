import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#subpunct 1

def simulare_poisson(lamb, nr_repetari):
    poisson_random = np.random.poisson(lam=lamb, size=nr_repetari) #cream distributia poisson
    print(poisson_random)
    return poisson_random

def randomized_poisson(nr_repetari):
    lamb_set=[1,2,5,10] #setul pt valorile lui lambda
    lamb = lamb_set[np.random.randint(0,len(lamb_set))] #alegem random un lambda din set
    poisson_random=np.random.poisson(lam=lamb,size=nr_repetari) #generam distributia poisson
    print(poisson_random)
    return poisson_random

poisson_1=simulare_poisson(1,1000)
print('\ntrecem la urm \n')
poisson_2=simulare_poisson(2,1000)
print('\ntrecem la urm \n')
poisson_3=simulare_poisson(5,1000)
print('\ntrecem la urm \n')
poisson_4=simulare_poisson(10,1000)


#subpunct 2
print('\n Poisson random: \n')
poisson_random=randomized_poisson(1000)

#subpunct 3

plt.hist(poisson_1, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = 1)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()



plt.hist(poisson_2, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = 2)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()



plt.hist(poisson_3, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = 5)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()




plt.hist(poisson_4, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = 10)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()




plt.hist(poisson_random, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = ales random)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

# in distributile cu lambda =1 si =2, mereu au mai putine valori au frecvente > 0 , frecventele pornind mereu de sus
# coborand pe masura ce valorile devin mai mari

#in distributiile cu lambda = 5 si = 10 cresterea in frecvente nu isi atinge maximul in primele valori cele mai mici
#si sunt mult mai putine valori ce au frecvente 0

#la distributia in care lambda e ales random, histograma se aseamana de mai multe ori cu cele in cazul in care lambda=5,10
#decat cu cele in cazul in care lambda=1,2 ,dar sunt si situatii in care histograma are "proprietati" din ambele situatii




#subpunctul c)
def randomized_poisson_alterat(nr_repetari):
    lamb=0
    alege_5=np.random.randint(0,2) #am crescut prob de a alege 5
    if alege_5==1:
        lamb=5
    else:
        lamb_set=[1,2,10] #setul pt valorile lui lambda
        lamb = lamb_set[np.random.randint(0,len(lamb_set))] #alegem random un lambda din set
    poisson_random=np.random.poisson(lam=lamb,size=nr_repetari) #generam distributia poisson
    print(poisson_random)
    return poisson_random

poisson_random_alterat=randomized_poisson_alterat(1000)
plt.hist(poisson_random, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (lambda = ales random cu prob lui 5 mai mare)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()
