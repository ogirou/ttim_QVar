# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 09:55:52 2022

@author: o.girou
"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from ttim import *
import pandas as pd
from scipy.special import exp1



# FONCTIONS ##

def DF2logSpace(df,lenLS):
    ts=np.logspace(np.log10(min(df.index).value/1e9), 
                   np.log10(max(df.index).value/1e9), 
                   lenLS)
    return ts


def paramCalage(df):
    txt=''
    for u in df.index:
        txt+=('{} = {:.0e}\n'.format(u,df['optimal'][u]))
    txt=txt.rstrip()   
    return txt   

    
# PARAM ENTREE ##

# entree des donnees de debit avec numpy
Q = np.loadtxt('synth_Q.csv', delimiter=';') # s ; m3/s
dQ=Q.transpose().copy()
# definition du delta Q pour superposition
dQ[1][1:]-=dQ[1][:-1]

# Q=pd.read_csv('PzP1_Q_tout_ET.csv', sep=';') #, encoding='latin1')
# Q['t (s)']=pd.to_timedelta(Q['t (s)'], 'S')
# Q=Q.set_index('t (s)')

# PARAM CALAGE ##

K = 1e-5    # m/s
b = 10      # m
T = K * b
Ss = 1e-3   # 1/m
S = Ss * b
r=0.026     # m

# temps de suivi
t_final=8e4 # s

# premier pas de temps / 1ere mesure (>= 1s)
t0=2000

# CALCUL ##

# transformation des donnees de delta Q en dataframe pandas
dQ_DF=pd.DataFrame(dQ.transpose())
# indexation des donnees
dQ_DF[0]=pd.to_timedelta(dQ_DF[0], 'S')
dQ_DF=dQ_DF.set_index(0)

# creation des pas de temps de clacul avec n logspaces suivant le nb de debits
t_Mod=np.logspace(np.log10(t0), np.log10(dQ[0][1]), 20)

try:
    for w in dQ[0][2:]:
        t_Mod=np.append(t_Mod, np.logspace(np.log10(t_Mod[-1]), 
                                          np.log10(w), 
                                          20)[1:])
except:
    pass

# ajout du dernier logspace avec la derniere mesure de rabattement    
t_Mod=np.append(t_Mod, np.logspace(np.log10(t_Mod[-1]), 
                                   np.log10(t_final), 
                                   20)[1:])
                 
# ajout de 0 en debut de serie pour le calcul des delta t
t_Mod=np.insert(t_Mod, 0, 0)

# transformation des donnees de t en dataframe pandas
t_DF=pd.DataFrame(t_Mod)
# indexation des donnees
t_DF[0]=pd.to_timedelta(t_DF[0], 'S')
t_DF=t_DF.set_index(0)


for v in range(len(dQ[1])):
    nom1='Q'+str(v)
    t_DF[nom1]=dQ[1][v]
    t_DF[nom1]["0 s":pd.Timedelta(dQ[0][v],"S")]=0 # PB
    nom2='t'+str(v)
    t_DF[nom2]=t_DF.index-pd.Timedelta(dQ[0][v],"S")

u=[[] for i in range(len(dQ[1]))]

for v in range(len(dQ[1])):
    nom1='Q'+str(v)
    nom2='t'+str(v)
    u[v]=(r**2 *S)/(4 *T * np.array(t_DF[nom2], dtype="float")/1e9) # PB

s=[[] for i in u]

for i in range(len(u)):
    nom1='Q'+str(i)
    s[i]=np.nan_to_num(np.array(t_DF[nom1])/(4*np.pi*T)*exp1(u[i]), nan=0)

s_final=np.sum(s, axis=0)
    

    


to1 = np.array(t_DF.index, dtype="float")/1e9 # temps (s)
ho1 = -s_final # rabattement (m) / '-' devant car valeurs positives de s

if to1[0] == 0 and ho1[0] == 0 :
    to1=to1[1:]
    ho1=ho1[1:]

ro1 = 0.026 # rayon d'observation (m)

# Pumping discharge

Qo = Q # m3/s
# Qo = np.loadtxt('PzP1_Q_descente.csv', delimiter=';') # m3/s

# Create model

KManu=1e-6

ml = ModelMaq(kaq=KManu, z=(12, 4), Saq=1e-4, tmin=1, tmax=t_final, phreatictop=True)
w = Well(ml, xw=0, yw=0, rw=ro1, tsandQ=Qo, layers=0)
ml.solve()



# Create calibration object, add parameters and first series. Fit the model. 
# The chi-square value is the mean of the squared residuals at the optimum.

cal = Calibrate(ml)
cal.set_parameter(name='kaq0', initial=0.00001)
cal.set_parameter(name='Saq0', initial=1e-4)
cal.series(name='obs1', x=ro1, y=0, layer=0, t=to1, h=ho1)
cal.fit(report=True)
display(cal.parameters) 
print('rmse:', cal.rmse())
print('mse:', cal.rmse() ** 2 * len(ho1))
h1a = ml.head(ro1, 0, to1, 0) # simulated head

# Calibration parameters extraction

calage=paramCalage(cal.parameters)
# calage+='\nkaq = {:.0e}'.format(KManu)

# Plot figures

plt.semilogx(to1, ho1, 'b.', label='observed 1')
plt.semilogx(to1, h1a[0], 'm-', label='model 1')
plt.title('calibrated well 1')
plt.xlabel('time (s)')
plt.ylabel('head (m)')
plt.text(min(to1),min(ho1),calage,bbox=dict(boxstyle='square', pad=0.4, 
                                            fc='w', ec='gray'))
#plt.text(min(to1),(min(ho1)+max(ho1))/2,calage,bbox=dict(boxstyle='square', pad=0.4, fc='w', 
#                                         ec='gray'))
plt.legend()
plt.grid('True')
plt.show()


# plt.loglog(to1, -ho1, 'b.', label='observed 1')
# plt.loglog(to1, -h1a[0], 'm.-', label='model 1')
# plt.title('calibrated well 1')
# plt.xlabel('time (s)')
# plt.ylabel('head (m)')
# plt.legend()
# plt.grid('True')
# plt.show()

