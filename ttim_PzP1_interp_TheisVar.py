# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:44:09 2021

@author: o.girou
"""



import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from ttim import *
import pandas as pd
from scipy.special import exp1

## Fonctions

# Extraction des paramètres de calage, ne fonction que pour le dataFrame
# cal.parameters de ttim

def paramCalage(df):
    txt=''
    for u in df.index:
        txt+=('{} = {:.0e}\n'.format(u,df['optimal'][u]))
    txt=txt.rstrip()   
    return txt   

def DF2logSpace(df,lenLS):
    ts=np.logspace(np.log10(min(df.index).value/1e9), 
                   np.log10(max(df.index).value/1e9), 
                   lenLS)
    return ts    

# Load data of the pumping well

drawdown = np.loadtxt('PzP1_s_tout.csv', delimiter=';')
# drawdown = np.loadtxt('PzP1_s_descente.csv', delimiter=';')

to1 = drawdown[:,0] # temps (s)
ho1 = -drawdown[:,1] # rabattement (m) / '-' devant car valeurs positives de s

if to1[0] == 0 and ho1[0] == 0 :
    to1=to1[1:]
    ho1=ho1[1:]

ro1 = 0.026 # rayon d'observation (m)

# Pumping discharge

Qo = np.loadtxt('PzP1_Q_tout.csv', delimiter=';') # m3/s
# Qo = np.loadtxt('PzP1_Q_descente.csv', delimiter=';') # m3/s

# Create model

KManu=4e-6
SManu=2

ml = ModelMaq(kaq=KManu, z=(12, 4), Saq=SManu, tmin=1, tmax=181770, phreatictop=False)
w = Well(ml, xw=0, yw=0, rw=ro1, tsandQ=Qo, layers=0)
ml.solve()



# Create calibration object, add parameters and first series. Fit the model. 
# The chi-square value is the mean of the squared residuals at the optimum.

cal = Calibrate(ml)
cal.set_parameter(name='kaq0', initial=KManu)
cal.set_parameter(name='Saq0', initial=SManu)
cal.series(name='obs1', x=ro1, y=0, layer=0, t=to1, h=ho1)
#cal.fit(report=True)
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
plt.text(min(to1),(min(ho1)+max(ho1))/2,calage,bbox=dict(boxstyle='square', pad=0.4, fc='w', 
                                         ec='gray'))
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



dQ=Qo.transpose().copy()
# definition du delta Q pour superposition
dQ[1][1:]-=dQ[1][:-1]

K = 4e-6    # m/s
b = 8      # m
T = K * b
Ss = 2   # 1/m
S = Ss * b
r=0.026     # m

# temps de suivi
t_final=max(to1) # s

# premier pas de temps / 1ere mesure
t0=1

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

t_final=np.array(t_DF.index, dtype="float")/1e9

plt.semilogx(to1, ho1, 'b.', label='observed 1')
plt.semilogx(to1, h1a[0], 'm-', label='model 1')
plt.semilogx(t_final, -s_final, 'c-', label='Theis 1')
plt.xlabel('time (s)')
plt.ylabel('head (m)')
plt.text(min(to1),(min(ho1)+max(ho1))/2,u"K={}m/s\nS={}".format(K,S),
bbox=dict(boxstyle='square', pad=0.4, fc='w',ec='gray'))
plt.legend()
plt.grid('True')
plt.show()


