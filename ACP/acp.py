#modification du dossier de travail
import os
os.chdir("D:/ACP")

#librairie pandas
import pandas

#version
#print(pandas_0.23.0_) # 0.23.0

#chargement de la première feuille de données
X = pandas.read_excel("data.xls",sheet_name=0,header=0,index_col=0)


#dimension
print(X.shape) # (18, 6)

#nombre d'observations
n = X.shape[0]
#nombre de variables
p = X.shape[1]
#affichage des données
print(X)

print("\n")
#scikit-learn
import sklearn

#classe pour standardisation
from sklearn.preprocessing import StandardScaler
#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(X)
print(Z)

print("\n")
#vérification - librairie numpy
import numpy
#moyenne
print(numpy.mean(Z,axis=0))

print("\n")
#écart-type
print(numpy.std(Z,axis=0,ddof=0))
print("\n")

#Nous sommes maintenant parés pour lancer l’ACP.
#classe pour l'ACP
from sklearn.decomposition import PCA
#instanciation
acp = PCA(svd_solver='full')
#affichage des paramètres
print(acp)

print("\n")
#calculs
coord = acp.fit_transform(Z)
#nombre de composantes calculées
print(" le nombre de composantes générées",acp.n_components_) # =p

print("\n")
#variance expliquée
print(" les valeurs propres, λk  associées aux axes factoriels sont :")
print(acp.explained_variance_)
print("\n")
print("\n")
#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print(eigval)
print("\n")

#ou bien en passant par les valeurs singulières
print(acp.singular_values_**2/n)

print("\n")

#proportion de variance expliquée
print(acp.explained_variance_ratio_)
print("\n")

#scree plot
import matplotlib.pyplot as plt
plt.plot(numpy.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()


#cumul de variance expliquée
plt.plot(numpy.arange(1,p+1),numpy.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()


print("Détermination du nombre de facteur à retenir")
#seuils pour test des bâtons brisés
bs = 1/numpy.arange(p,0,-1)
bs = numpy.cumsum(bs)
bs = bs[::-1]


print("\n")

#test des bâtons brisés
print(" les valeurs propres et leurs seuils : ")
print(pandas.DataFrame({'Val.Propre':eigval,'Seuils':bs}))


#Représentation des individus
print("Représentation des individus")
#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6) #même limites en abscisse
axes.set_ylim(-6,6) #et en ordonnée
#placement des étiquettes des observations
for i in range(n):
 plt.annotate(X.index[i],(coord[i,0],coord[i,1]))
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
#affichage
plt.show()

#Qualité de représentation – Les COS² (cosinus carré)
#contribution des individus dans l'inertie totale
di = numpy.sum(Z**2,axis=1)
print(pandas.DataFrame({'ID':X.index,'d_i':di}))
print("\n")
#qualité de représentation des individus - COS2
cos2 = coord**2
for j in range(p):
 cos2[:,j] = cos2[:,j]/di
print(pandas.DataFrame({'id':X.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]}))
print("\n")

#vérifions la théorie - somme en ligne des cos2 = 1
print(numpy.sum(cos2,axis=1))
print("\n")


#Contribution des individus aux axes (CTR).
print("Contribution des individus aux axes (CTR). ")
#contributions aux axes
ctr = coord**2
for j in range(p):
 ctr[:,j] = ctr[:,j]/(n*eigval[j])

print(pandas.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]}))
#vérifions la théorie
print(numpy.sum(ctr,axis=0))

print("\n")

#Représentation des variables
print("Représentation des variables ")
#le champ components_ de l'objet ACP
print(acp.components_)
print("\n")
#racine carrée des valeurs propres
sqrt_eigval = numpy.sqrt(eigval)
#corrélation des variables avec les axes
corvar = numpy.zeros((p,p))
for k in range(p):
 corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

#afficher la matrice des corrélations variables x facteurs
print(corvar)
print("\n")
#on affiche pour les deux premiers axes
print(pandas.DataFrame({'id':X.columns,'COR_1':corvar[:,0],'COR_2':corvar[:,1]}))
print("\n")
#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage
plt.show()


#Qualité de représentation des variables (COS²)
print("Qualité de représentation des variables (COS²)")

#cosinus carré des variables
cos2var = corvar**2
print(pandas.DataFrame({'id':X.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))

#Contribution des variables aux axes (CTR)
print("Contribution des variables aux axes (CTR)")
#contributions
ctrvar = cos2var
for k in range(p):
 ctrvar[:,k] = ctrvar[:,k]/eigval[k]
#on n'affiche que pour les deux premiers axes
print(pandas.DataFrame({'id':X.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))

