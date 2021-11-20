"""
Simulation informatique du mouvement 3D d'une bille sur des rails.
Groupe 11.22 : Debelle Thomas, Debouvrie Elian, Debroux Bastien, de Woot Grégoire, Katsani Lara et Koch Raphaëlle.
Version du 20-11-21
UCLouvain, Cours de projet 1, LEPL1501, année 2021-2022
"""

#Import des bibliothèques et programmes annexes 
import path3d as p3d                      #Programme annexe : fornis par les professe
import numpy as np                        #Package numpy
import matplotlib.pyplot as plt           #Package matplotlib
from mpl_toolkits.mplot3d import Axes3D

#Paramètres physiques du circuit
g = np.array([0,0, -9.81])       #Constante de gravité
m = 0.005                        #Masse de la bille
r = 0.8                          #Rayon de la bille
b = 0.013                        #Ecartement des rails
h = np.sqrt(r**2-(b**2/4))       #Distance entre les rails et le contre de gravité de la bille
e = 0.004                        #Paramètre de frottement calculé grâce à la simulation 2D

xyzPoints = np.loadtxt('looping_points.txt', unpack=True, skiprows=1)    #Chargement des points de passages

path = p3d.path(xyzPoints)

sPath, X, T, C = path
sPath_geo, xyzPath_geo, TPath_geo, CPath_geo = path



#----------Affichage du circuit grâce à matplolib----------#
num = 30                         # nombre de jalons
length = sPath_geo[-1]
sMarks = np.linspace(0, length, num)
xyzMarks = np.empty((3, num))    # coordonnées
TMarks = np.empty((3, num))      # vecteur tangent
CMarks = np.empty((3, num))      # vecteur de courbure

for i in range(num):
    xyz = p3d.ainterp(sMarks[i], sPath_geo, xyzPath_geo)
    T = p3d.ainterp(sMarks[i], sPath_geo, TPath_geo)
    C = p3d.ainterp(sMarks[i], sPath_geo, CPath_geo)

    xyzMarks[:,i] = xyz
    CMarks[:,i] = C
    TMarks[:,i] = T
    
# graphique 3D : points, chemin et vecteurs
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect(np.ptp(xyzPath_geo, axis=1))
ax.plot(xyzPoints[0],xyzPoints[1],xyzPoints[2],'bo', label='points')
ax.plot(xyzPath_geo[0],xyzPath_geo[1],xyzPath_geo[2], 'k-', lw=0.5, label='path')    
scale = 0.5*length/num
ax.quiver(xyzMarks[0],xyzMarks[1],xyzMarks[2],
          scale*TMarks[0],scale*TMarks[1],scale*TMarks[2],
          color='r', linewidth=0.5, label='T')
ax.quiver(xyzMarks[0],xyzMarks[1],xyzMarks[2],
          scale*CMarks[0],scale*CMarks[1],scale*CMarks[2],
          color='g', linewidth=0.5, label='C')
ax.legend()            
plt.show()


#----------Simulation----------#

#paramètres de la simulation
tEnd = 10      #Durée de la simulation en secondes 
dt = 0.1       #Durée entre chaque prise de valeure 

#Initialisation des variables de simulation
steps = int(tEnd / dt)         # nombre de pas de la simulation
tSim = np.zeros(steps+1)       # temps: array[steps+1] * [s]
sSim = np.zeros(steps+1)       # distance curviligne: array[steps+1] * [m]
VsSim = np.zeros(steps+1)      # vitesse tangentielle: array[steps+1] * [m/s]
xSim = np.zeros(steps+1)       # coordonnées x par lesquelles passe la bille
ySim = np.zeros(steps+1)       # coordonnées y
zSim = np.zeros(steps+1)       # coordonnées z
TSim = np.zeros(steps+1)       # valeurs du vecteur unitaire tangent
CSim = np.zeros(steps+1)       # valeurs du vecteur unitaire radial
As = np.zeros(steps+1)         # valeurs de l'accélération curviligne

M = 1 + 2/5*r**2/h**2          # coefficient d'inertie [1]

# valeurs initiales:
tSim[0] = 0
sSim[0] = 0
VsSim[0] = 0

i = 0

with open("Donnes_simulation_.txt", "w") as file:
    file.write("tSim \t VsSim \t sSim[i] \t\t coordonnées \n")
    
# boucle de simulation:
while i < steps:
    X, T, C = p3d.path_at(sSim[i], path)
    gs = np.dot(g, T)
    gsT = gs * T
    gn = g - gsT
    VsC = (VsSim[i]**2) * C
    Gn = VsC - gn
    Gn = np.linalg.norm(Gn)

    As = (gs - ((e*VsSim[i])/h) * Gn)/(1+((2*r**2)/(5*h**2)))
    
    xSim[i] = X[0]
    ySim[i] = X[1]
    zSim[i] = X[2]
    
    VsSim[i+1] = VsSim[i] + As * dt
    sSim[i+1] = sSim[i] + VsSim[i+1] * dt
    tSim[i+1] = tSim[i] + dt
    
    with open("Donnes_simulation_.txt", "a") as file:
        file.write("{0:.3f} \t {1:.3f} \t {2:.3f} \t ({3:.2f},{4:.2f},{5:.2f}) \n".format(tSim[i], VsSim[i], sSim[i], xSim[i], ySim[i], zSim[i]))
    i = i+1
    
with open("Donnes_simulation_.txt", "a") as file:
    file.write("{0:.3f} \t {1:.3f} \t {2:.3f} \t ({3:.2f},{4:.2f},{5:.2f}) \n".format(tSim[i], VsSim[i], sSim[i], xSim[i], ySim[i], zSim[i]))

    
# plot distance et vitesse et hauteur
plt.figure()
plt.subplot(311)
plt.plot(tSim, sSim, label='s')
plt.ylabel('s [m]')
plt.xlabel('t [s]')
plt.subplot(312)
plt.plot(tSim, VsSim, label='vs')
plt.ylabel('Vs [m/s]')
plt.xlabel('t [s]')
plt.subplot(313)
plt.plot(tSim, zSim, label='z')
plt.ylabel('z [m]')
plt.xlabel('t [s]')
plt.show()

EpSim = 9.81*zSim*m # énergie potentielle spécifique [m**2/s**2]
EkSim = 0.5*m*VsSim**2  # énergie cinétique spécifique [m**2/s**2]

# plot énergies
plt.figure()
plt.plot(tSim, EpSim, 'b-', label='Ep/m')
plt.plot(tSim, EkSim, 'r-', label='Ek/m')
plt.plot(tSim, EkSim+EpSim, 'k-', label='E/m')
plt.legend()
plt.ylabel('Energy/mass [J/kg]')
plt.xlabel('t [s]')
plt.show()