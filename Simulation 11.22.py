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
#from mpl_toolkits.mplot3d import Axes3D

#Paramètres physiques du circuit
g = np.array([0,0, -9.81])       #Constante de gravité
m = 0.005                        #Masse de la bille
r = 0.008                        #Rayon de la bille
b = 0.013                        #Ecartement des rails
h = np.sqrt(r**2-(b**2/4))       #Distance entre les rails et le contre de gravité de la bille
e = 0.0004                       #Paramètre de frottement calculé grâce à la simulation 2D note:rajouter 2 0 ??
M = (1+((2*r**2)/(5*h**2)))      # coefficient d'inertie [1]
bloque = False                   #Pour savoir si la bille fait bien tout le circuit
timebloque = 0                   #Temps où la bille bloque

xyzPoints = np.loadtxt("C:\\Users\\thoma\\OneDrive\\Bureau\\Circuit irl.txt", unpack=True, skiprows=1)    #Chargement des points de passages

path = p3d.path(xyzPoints)       #création du tracée à partir des points fournis

sPath, X, T, C = path            #attribution des valeurs à chacune des variables
sPath_geo, xyzPath_geo, TPath_geo, CPath_geo = path #on répète afin de réaliser différentes oppérations



#----------Affichage du circuit grâce à matplol ib----------#
num = 30                         # nombre de jalons
length = sPath_geo[-1]           # longueur circuit déroulé
sMarks = np.linspace(0, length, num) #subdivision de la longueur en n parties égales
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
scale = 0.03*length/num
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
tEnd = 100     # Durée de la simulation en secondes
dt = 0.001     # Durée entre chaque prise de valeure

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

# valeurs initiales:
tSim[0] = 0
sSim[0] = 0
VsSim[0] = 0

i = 0                           # valeur qui va permettre de suivre l'évolution tout les i temps
bloquetext = ""                 # création d'un texte vide qui se remplira si notre bille n'avance plus

with open("Donnes_simulation_.txt", "w") as file:
    file.write("tSim \t VsSim \t sSim[i] \t As\t\t coordonnées \n") # écriture de l'entête de notre fichier

# boucle de simulation:
while i < steps:
    X, T, C = p3d.path_at(sSim[i], path)
    gs = np.dot(g, T)           # utilisation des formules fournies dans les documents de physique
    gsT = gs * T
    gn = g - gsT
    VsC = (VsSim[i]**2) * C
    Gn = VsC - gn
    Gn = np.linalg.norm(Gn)

    As = (gs - (((e*VsSim[i])/h) * Gn))/(M) # formule de l'accélération
    #As = (5*h*(h*gs-e*VsSim[i]*Gn))/((2*r**2 + 5*h**2))

    xSim[i] = X[0]
    ySim[i] = X[1]
    zSim[i] = X[2]

    VsSim[i + 1] = VsSim[i] + As * dt  # formule de la vitesse
    sSim[i + 1] = sSim[i] + VsSim[i+1] * dt    # formule pour l'accélération curviligne
    tSim[i + 1] = tSim[i] + dt  # avancement du temps

    if i > 1:
        if xSim[i] == xSim[i-1] and ySim[i] == ySim[i-1] and zSim[i] == zSim[i-1]: # si la bille reste au même endroit on écourte la simulation en temps
            xSim = xSim[:i]
            ySim = ySim[:i]
            zSim = zSim[:i]
            VsSim = VsSim[:i]
            tSim = tSim[:i]
            sSim = sSim[:i]
            break
        elif zSim[i] < zSim[i-1]:           # trouver le point le plus bas pour mettre à jour le référentiel pour l'énergie potentielle gravifique
            lowzsim = zSim[i]

    if not bloque and sSim[i] < sSim[i - 1]:# pour voir si la bille n'est pas bloqué
        bloque = True
        timebloque = i * dt
        bloquetext = "la bille est bloquée en {} s".format(timebloque)

    if i+1 == steps: # pour prendre en compte le cas steps
        xSim[i+1] = X[0]
        ySim[i+1] = X[1]
        zSim[i+1] = X[2]

    with open("Donnes_simulation_.txt", "a") as file: # enregistrement des données dans un fichier texte
        file.write("{0:.3f} \t {1:.3f} \t {2:.3f} \t {3:.3f}\t ({4:.2f},{5:.2f},{6:.2f}) \n".format(tSim[i], VsSim[i], sSim[i],As, xSim[i], ySim[i], zSim[i]))

    i += 1

# plot distance et vitesse et hauteur
plt.figure()
plt.subplot(311)
plt.plot(tSim, sSim, label='s')
plt.ylabel('s [m]')
plt.xlabel('t [s]')
plt.text(timebloque,0, bloquetext) # écrit sur le graphique au moment où la bille est bloquée avec le temps "testez avec looping_points1.txt pour voir"
plt.subplot(312)
plt.plot(tSim, VsSim, label='vs')
plt.ylabel('Vs [m/s]')
plt.xlabel('t [s]')
plt.subplot(313)
plt.plot(tSim, zSim, label='z')
plt.ylabel('z [m]')
plt.xlabel('t [s]')
plt.show()

EpSim = 9.81*(zSim-lowzsim) # énergie potentielle spécifique [m**2/s**2]
EkSim = 0.5*M*VsSim**2  # énergie cinétique spécifique [m**2/s**2]

# plot énergies
plt.figure()
plt.plot(tSim, EpSim, 'b-', label='Ep/m')
plt.plot(tSim, EkSim, 'r-', label='Ek/m')
plt.plot(tSim, EkSim+EpSim, 'k-', label='E/m')
plt.legend()
plt.ylabel('Energy/mass [J/kg]')
plt.xlabel('t [s]')
plt.show()
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect(np.ptp(xyzPath_geo, axis=1))
ax.plot(xSim,ySim,zSim,'bo', label='points')
scale = 0.5*length/num

ax.legend()
plt.show()
"""