# YOLOv4_attendance

Script de détection automatique d'objets (ici les "humains") dans des images, basé sur le modèle [YOLOv4](https://github.com/AlexeyAB/darknet) entrainé sur le jeu de données [COCO](https://cocodataset.org/#home).

## Description

Ce script utilise le modèle de réseau de neurones [yolov4](https://github.com/AlexeyAB/darknet) pour détecter des objets dans des images provenant d'un serveur FTP ou d'un répertoire local.

Ce script permet ainsi de compter automatiquement, sans visualisation par l'utilisateur, le nombre de personnes présentes sur des images, notamment dans le cadre de suivis de la fréquentation réalisés avec des pièges photos à déclenchement automatique. Le script compte le nombre maximum d'humains au sein de chaque séquence, retenu comme taille de groupe.

En complément, voir le [rapport de stage](https://data.ecrins-parcnational.fr/documents/stages/2023-09-rapport-stage-Aurelien-Coste-photos-IA-frequentation.pdf) d'Aurélien Coste qui a travaillé en 2023 sur la version 0.2.0, ainsi que son [support de restitution](https://data.ecrins-parcnational.fr/documents/stages/2023-09-restitution-stage-Aurelien-Coste-photos-IA-frequentation.pdf).

## Installation

Commencez par cloner le dépôt git.

```
git clone git@github.com:PnEcrins/yolov4-attendance.git
```

Ensuite, vous avez besoin de récupérer le fichier des paramètres du modèle et de le déposer dans le répertoire `/model`. Il est disponible à cette adresse :
[:link: Modèle](http://mathieu.garel.free.fr/yolo/yolov4.h5)

Une fois le modèle récupéré, il vous faut créer votre version du fichier de configuration.

```
cp config_sample.json config_json
```

#### Description des paramètres de configuration

- **ftp_server :** Nom du serveur FTP  
  Si vous ne voulez pas utiliser de FTP, il faut laisser le champ vide (`""`) 
  Dans ce cas, la classification se fera via le répertoire local indiqué dans le paramètre `local_folder`  
- **ftp_username :** Username pour la connexion au serveur FTP  
- **ftp_password :** Mot de passe pour la connexion au serveur FTP  
- **ftp_directory :** Répertoire contenant les images sur le serveur FTP  
- **local_folder :** En mode FTP, il s'agit du répertoire dans lequel les images seront téléchargées puis classifiées puis supprimées.   
  En mode local, il s'agit du répertoire contenant les images à classifier  
- **output_folder :** Répertoire dans lequel les fichiers de sortie seront stockés  
- **model_file :** Chemin d'accès au fichier .h5 du modèle  
- **treshold :** Valeur du seuil de classification pour le modèle. Cette valeur varie de 0 à 1. Plus la valeur est basse, plus nous sommes permissifs avec les classifications. Plus la valeur est haute, plus nous sommes restrictifs avec les classifications.  
- **sequence_duration :** Valeur (en secondes) du temps de séquence. Le temps de séquence est utilisé par le script pour compter les groupes d'individus. Lors de la classification de l'image n, si l'image n-1 a été classifiée il y a moins du temps de séquence choisi alors le script considère qu'il s'agit du même groupe d'individus et donc il ne compte pas deux fois ce groupe.  
  La valeur de base est de 10 secondes. Selon la fréquentation de votre sentier, vous pouvez baisser jusqu'à 5 s'il est très fréquenté et monter jusqu'à 15 s'il est très peu fréquenté. Au-delà de cet intervalle, les résultats sont moins bons.  
- **time_step :** Pas de temps pour concaténer les classifications du modèle et sortir un fichier avec un nombre de passage en fonction du pas de temps choisi.  
  Valeur de base : 'Hour', peut prendre les valeurs : 'Day', 'Month' et 'Year'  
- **output_format :** Format du fichier de sortie.  
  Valeur de base 'csv', peut prendre les valeurs : 'dat'  
- **csv_column :** Modifie le mapping des colonnes dans le fichier de sortie  

Une fois le fichier de configuration modifié selon vos besoins, vous pouvez créer un environnement virtuel python :

```
python3 -m virtualenv venv
source venv/bin/activate
```

Ensuite, vous pouvez installer les librairies nécessaires au fonctionnement du code via la commande :

```
pip install -e .
```

## Utilisation

Pour lancer le script, exécuter :

```
python3 yolov4_attendance.py
```

N'oubliez pas de créer/modifier votre fichier de config !

## Auteurs

* Mathieu Garel (OFB)
* Aurélien Coste (Parc national des Ecrins)
* Théo Lechémia (Parc national des Ecrins)
