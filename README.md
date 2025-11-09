# Movie-Recommendation-System
## 1. Objectif

---
## 2. Dataset
Nous utilisons le dataset **MovieLens 100k** pour entraîner notre système de recommandation.

- **Fichiers principaux :**
  - `u.data` : notes des utilisateurs pour les films
  - `u.item` : informations sur les films (titre, date, genres)
  - `u.user` : informations sur les utilisateurs (âge, sexe, occupation)
- **Source :** [MovieLens 100k Dataset]([https://grouplens.org/datasets/movielens/100k/](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/versions/1))
---
## 3. Méthodes de recommandation explorées
### 3.1 Matrix Factorization (MF)
**Définition :**
La Matrix Factorization est une approche largement utilisée dans les systèmes de recommandation, particulièrement lorsqu’on travaille avec une matrice **Utilisateur x Item** contenant des notes, souvent incomplète (sparse).  
L’idée principale est de **décomposer** cette matrice en deux matrices de plus petite dimension :
- **U** : matrice des *représentations latentes des utilisateurs*
- **V** : matrice des *représentations latentes des items*
Ainsi, la prédiction de la note qu’un utilisateur donnera à un item est obtenue par le **produit scalaire** entre leurs vecteurs latents :

$$
\hat{R}_{u,i} = U_u \cdot V_i
$$

Cette méthode permet de :
- Réduire la dimension des données
- Capturer des relations implicites (goûts, préférences, similarités)
- Gérer efficacement les matrices très creuses
En pratique, l’apprentissage consiste à **ajuster U et V** de manière à minimiser l’erreur entre les notes prédites et les notes réelles, généralement via **descente de gradient** :
$$
\min_{U,V} \sum_{(u,i) \in known} (R_{u,i} - U_u \cdot V_i)^2 + \lambda (||U||^2 + ||V||^2)
$$

où **λ** est un terme de régularisation pour éviter le sur-apprentissage.

![Matrix Factorization](MF.jpg)

**Avantages :**
- Performant pour des grands ensembles de données
- Représentation compacte et significative des préférences
**Limites :**
- Suppose que les préférences peuvent être capturées linéairement
- Ne tient pas compte du contenu (ex : description des films, profils utilisateurs) sans extension

**Paramètres d’entraînement :**  
- **Dimension des embeddings** : 20  
- **Optimiseur** : Adam, learning rate = 0.01  
- **Fonction de perte** : Mean Squared Error (MSE)  
- **Early stopping** : le modèle arrête l’entraînement si la RMSE sur le jeu de validation ne s’améliore pas pendant 3 epochs consécutives.  

**Résultats :**  
- RMSE sur le jeu de test : **0.0817**, indiquant une bonne capacité du modèle à prédire les notes.  
- Exemple de recommandations générées pour un utilisateur : les 10 films les mieux notés qu’il n’a pas encore vus, comme *Pushing Hands (1992)*, *Other Voices, Other Rooms (1997)*, etc.  
**Résumé :**  
MF permet de transformer le problème de recommandation en apprentissage de vecteurs latents pour utilisateurs et films, ce qui facilite la prédiction de notes et la génération de recommandations personnalisées. C’est une méthode simple mais puissante, qui sert souvent de base pour des modèles plus avancés comme NCF ou les approches par graphes.
---
### 3.2. Neural Collaborative Filtering (NCF)
**Définition :**
**Avantages :**
**Limites :**
**Résultats :** 
**Paramètres d’entraînement :** 
**Résumé :** 
---
### 3.3. GraphSAGE (GNN)
**Définition :**
**Avantages :**
**Limites :**
**Résultats :** 
**Paramètres d’entraînement :** 
**Résumé :**
---

### 5.2. Résultats NCF
### 5.3. Résultats GraphSAGE
## 6. Exemples de recommandations générées
## 7. Améliorations possibles & Perspectives
