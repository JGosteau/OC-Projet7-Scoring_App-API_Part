# OC-Projet7-Scoring_App-API_Part
Projet 7 du parcours Data Scientist de OpenClassroom - Partie serveur web API

Nous souhaitons mettre en place un outil de « scoring credit » pour la société « Prêt à dépenser », afin de déterminer la probabilité de remboursement d’un client, dans l’objectif de classifier la demande en crédit en accordé ou refusé. Pour définir le modèle de classification, la société nous a fourni une base de données de plus de 300 000 clients anonymisés, comportant entre autres des informations comportementales, les historiques de demande de crédits dans diverses institutions financières. Nous nous retrouverons avec plus de 300 variables. 

Ce repository contient les éléments nécessaires au fonctionnement d’un serveur api faisant appel aux modèles de classification entrainé sur le jeu de données fournis (disponible à l’adresse suivante : https://www.kaggle.com/competitions/home-credit-default-risk/data).

Les fonctions permettant d’entrainer et enregistrer le modèle sont aussi disponibles dans ce repository.

Le repository contient les éléments suivants :
-	api/api.py : gère le serveur api
-	model/models.py : définition de la classe du modèle
-	model/traintest.py : définition des fonctions permettant le feature engineering des données et la création de données train/test.
-	model/data : contient les données nécessaires au fonctionnement du serveur.
-	model/saved_models : contient les modèles entrainés.
-	model/utils/filtering.py : outil de filtrage des données.
-	model/utils/preprocessing.py : outil pour le feature engineering des données.
-	model/utils/multi_label_encoder.py : définition de la classe MultiLabelEncoder permettant de gérer plusieurs LabelEncoder en même temps.
-	requirements.txt : liste les librairies python requises pour le fonctionnement des programmes.
-	notebook.ipynb : Notebook jupyter présentant la création du modèle.
