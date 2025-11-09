# Movie-Recommendation-System
## 1. Objectif

L'objectif de ce projet est de **développer un système de recommandation de films** capable de prédire les notes que les utilisateurs pourraient attribuer à des films qu'ils n'ont pas encore vus, et de leur fournir des recommandations personnalisées.  

Pour cela, nous explorons plusieurs méthodes de filtrage collaboratif :  
- **Matrix Factorization (MF)** : approche linéaire basée sur la factorisation de la matrice utilisateur-item.  
- **Neural Collaborative Filtering (NCF)** : modèle non-linéaire basé sur des réseaux de neurones pour capturer les interactions complexes.  
- **GraphSAGE (GNN)** : approche par graphes qui exploite la structure relationnelle des interactions utilisateurs-items pour générer des recommandations plus riches et contextuelles.  

Ce projet permet de comparer ces méthodes sur un **dataset réel (MovieLens 100k)** et d’analyser leur capacité à fournir des recommandations pertinentes.

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
Neural Collaborative Filtering (NCF) est un modèle de **filtrage collaboratif basé sur les réseaux de neurones**.  
Il utilise des **embeddings pour les utilisateurs et les items** et passe la concaténation de ces embeddings dans un **Multi-Layer Perceptron (MLP)** pour prédire les notes ou préférences des utilisateurs.

![Neural Collaborative Filtering](NCF.jpg)

**Avantages :**
- Permet de **capturer des interactions complexes non linéaires** entre utilisateurs et items, contrairement aux modèles linéaires classiques (ex: matrix factorization).  
- Flexible : la profondeur et largeur du MLP peuvent être ajustées selon la taille et la complexité des données.  
- Bonne performance sur des datasets avec un **grand nombre d’utilisateurs et d’items**.

**Limites :**
- **Exigeant en ressources** (GPU recommandé pour de grands datasets).  
- Risque de **surapprentissage** si le dataset est petit et le réseau trop profond.  
- Nécessite une **phase d’entraînement longue** pour converger correctement.  
- La **cold-start problem** (nouveaux utilisateurs ou items) reste un défi.

**Résultats :**
- Sur le dataset utilisé (ex: MovieLens 100k) :  
  - RMSE final : environ **0.21 - 0.28**  
  - MAE final : environ **0.16 - 0.23**  
- Le modèle converge rapidement, avec une légère surperformance sur les données d’entraînement par rapport à la validation.  
- Les recommandations générées pour un utilisateur sont **cohérentes avec ses préférences passées**.

**Paramètres d’entraînement :**
- **Optimizer** : Adam  
- **Learning rate** : 0.001  
- **Batch size** : 2048  
- **Epochs** : 50  
- **Loss function** : MSE (Mean Squared Error)  
- **Scheduler** : StepLR (γ=0.1 tous les 10 epochs)  
- **Dropout** : 0.3  
- **Embedding size** : 64  
- **MLP hidden layers** : (128, 64, 32)  

**Résumé :** 
NCF est un modèle puissant pour la recommandation personnalisée, capable de modéliser les relations complexes entre utilisateurs et items.  
Il est particulièrement efficace pour des datasets de taille moyenne à grande, mais nécessite un entraînement sur GPU et une régularisation adaptée pour éviter le surapprentissage.  
Les résultats montrent qu’il peut fournir des recommandations pertinentes tout en maintenant une bonne précision prédictive.

---
### 3.3. GraphSAGE (GNN)
**Définition :**  
GraphSAGE (Graph Sample and AggregatE) est un modèle de Graph Neural Network (GNN) qui apprend des représentations (embeddings) de nœuds en échantillonnant et en agrégeant les caractéristiques de leurs voisins dans un graphe. Ici, il est utilisé pour la recommandation de films en représentant les utilisateurs et les films comme des nœuds et les interactions (notes) comme des arêtes.

![GraphSAGE](GraphSAGE.jpg)

**Avantages :**  
- Capable de généraliser aux nœuds non vus grâce à l'agrégation de voisinage.  
- Exploite efficacement la structure du graphe (relations utilisateurs ↔ films).  
- Permet de générer des embeddings riches pour la prédiction et la recommandation.  

**Limites :**  
- Peut être coûteux en mémoire et en calcul pour des graphes très grands.  
- La performance dépend fortement de la qualité et de la densité des interactions dans le graphe.  
- Requiert un ajustement des hyperparamètres (taille des embeddings, nombre de couches, learning rate…).  

**Résultats :**  
- Test RMSE : 0.2811  
- Test MAE : 0.2326   

**Paramètres d’entraînement :**  
- Embedding initial : dimension 64  
- Couche GraphSAGE : 2 couches, 64 unités chacune  
- Optimizer : Adam, lr = 0.001  
- Loss : MSE  
- Epochs : 10  

**Résumé :**  
GraphSAGE a permis de construire un modèle de recommandation basé sur le graphe des interactions utilisateurs/films. Les métriques RMSE et MAE montrent que le modèle prédit correctement les notes normalisées et génère des recommandations pertinentes, exploitant la structure relationnelle du graphe.


