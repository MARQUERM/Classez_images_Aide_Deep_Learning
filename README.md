# Classement des images de chiens avec des algorithmes de Deep Learning: OCR
Ce projet a été réalisé en partenariat avec l'association de protection des animaux Le Refuge. L'objectif était de développer un algorithme capable de classer automatiquement les images de chiens en fonction de leur race, afin d'accélérer le processus d'indexation de la base de données de l'association.

# Contexte
L'association Le Refuge dispose d'une grande quantité d'images de chiens non indexées sur leurs disques durs. Ils ont sollicité mon expertise en tant que Data Analyst pour développer un modèle de classification d'images afin de faciliter leur travail d'indexation.

# Données
Pour entraîner mon algorithme, j'ai utilisé le Stanford Dogs Dataset, une base de données contenant des images de différentes races de chiens.

# Méthodologie
Pour répondre au besoin de l'association, j'ai mis en œuvre deux approches basées sur des réseaux de neurones convolutionnels (CNN) :

1. Création d'un modèle CNN personnalisé en utilisant des techniques de data augmentation et d'optimisation des hyperparamètres.
2. Utilisation du transfer learning avec un réseau déjà entraîné, en adaptant les dernières couches pour prédire les classes de races de chiens qui nous intéressent.

# Livrables
Un notebook contenant l'analyse et le prétraitement des images.
Un notebook décrivant la création et l'entraînement du modèle CNN personnalisé, ainsi que les simulations des différentes valeurs des hyperparamètres.
Un notebook détaillant l'entraînement des modèles basés sur le transfer learning.
Un programme de prédiction exécuté en local, qui prend des images de chiens en entrée et retourne les races réelles et prédites des chiens.