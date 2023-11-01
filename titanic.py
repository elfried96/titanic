# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:39:20 2023

@author: KINZOUN Elfried
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic = titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0,inplace= True)
titanic['sex'].replace(['male','female'],[0,1],inplace = True)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

y = titanic['survived']
x = titanic.drop('survived',axis=1)

model.fit(x,y)
model.score(x,y)

model.predict(x)


nom = input ("Quelle est votre nom ?")
prenom = input ("Quelle est votre prenom")
def survie(model, pclass, sex, age):
    x = np.array([pclass, sex, age]).reshape(1, 3)
    prediction = model.predict(x)
    probability = model.predict_proba(x)
    return prediction, probability

# Vous devez charger votre modèle ici en utilisant la syntaxe appropriée.
# model = charger_modele()

while True:
    reponse = input('Voulez-vous savoir si vous seriez l un des survivants du Titanic (oui/non) ? ')
    
    if reponse.lower() == 'oui':
        pclass = int(input("En quelle classe voyageriez-vous ? (1, 2, 3) : "))
        sex = int(input("Quel est votre sexe ? (0 pour les hommes, 1 pour les femmes) : "))
        age = int(input("Quel est votre âge ? : "))
        # Vous devez charger votre modèle ici pour qu'il fonctionne.
        # model = charger_modele()
        
        # Exemple d'utilisation du modèle
        # Remplacez les valeurs par les données de l'utilisateur.
        prediction, probability = survie(model, pclass, sex, age)
        if prediction == 0:
            print(f" M/Mr  {nom} {prenom} Vous ne survivrez probablement pas au Titanic.")
        else:
            print("Bravo, vous avez de bonnes chances de survivre au Titanic.")
        
        print("Probabilité : ", probability)
    else:
        break
        
    break
