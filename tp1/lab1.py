import numpy as np
# import matplotlib.pyplot as plt

# les arrays sont batis avec les dimensions suivantes:
# pluie , arroseur , watson , holmes
# et chaque dimension : faux , vrai

# array de dimension 4 (think of a tree)
prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))

prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Pr(Arroseur)={}\n".format(np.squeeze(prob_arroseur)))

watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print("Pr(Watson|Pluie)={}\n".format(np.squeeze(watson)))

# 1 ere dimension npluie  pluie
# 2 eme dimension narro arro
# 3 eme dimension nholmes holmes
# in the reshape take into account that we skip the dim for watson
holmes = np.array([[[1,0], [0.1,0.9]], [[0, 1], [0, 1]]]).reshape(2, 2, 1, 2)
print("Pr(Holmes|Pluie ,arroseur)={}\n".format(np.squeeze(holmes)))

watson[0, :, 1, :]  # prob watson mouille − pluie
(watson * prob_pluie).sum(0).squeeze()[1]  # prob gazon watson mouille
holmes[0, 1, 0, 1]  # prob gazon holmes mouille si arroseur − pluie

compute = (holmes * prob_arroseur * prob_pluie) # sum of proba is = 1

# P(h =1) = P(holmes sachant a = 1, p = 1) p(p=1) p(a=1) + p(h=1 sachant a=0, p=1) p(p=1) p(a=0)
pholmes = compute[:, :, :, 1].sum() #verified

# P(A=1 sachant H=1)
# use bayes rules : P(H sachant A) P(A) / P(H)
parro = np.squeeze(prob_arroseur)[1]
# pholmes is res of prev q
pholmesGarro = compute[:, 1, :, 1].sum() #verified
parroGholmes = pholmesGarro * parro / pholmes

# it differs from the raw P(A) in the sense that it takes into account that holmes = 1
# it is like a subset of the world (because we "rebase" our probability on the fact that holmes = 1) that we read from the bottom up

# P(A=1 sachant H=1 et W=1)
# first we need the global probability table
globalp = holmes * watson * prob_arroseur * prob_pluie
# we already got P(A) as parro
pholmesAwatson = globalp[:, :, 1, 1].sum() # verified

pholmesAwatsonGarro = globalp[:, 1, 1, 1].sum()/parro # verified

parroGholmesAwatson = pholmesAwatsonGarro * parro / pholmesAwatson

# the diff between P(A given H, W) and P(A given H) comes from the low
# probability of watson = 1 while holmes = 1 outside from p = 1
