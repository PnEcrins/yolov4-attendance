# YoloV4_attendance

Script de classification automatique d'images basé sur le modèle [yolov4](https://github.com/AlexeyAB/darknet).

## Description

Ce script utilise le modèle de classification [yolov4](https://github.com/AlexeyAB/darknet) pour classifier des images provenant d'un serveur FTP ou d'un répertoire local. À savoir que le modèle yolov4 a été re-entrainé par Mathieu Garel de l'Office français de la biodiversité pour le rendre plus performant dans le suivi de la fréquentation.

## Installation

Commencez par cloner le dépôt git.

```
git clone git@github.com:PnEcrins/yolov4-attendance.git
```

Ensuite, vous avez besoin de récupérer le fichier des paramètres du modèle re-entrainé par Mathieu Garel et de le déposer dans le répertoire *model*. Il est disponible à cette adresse :
[Modèle réentrainé](http://mathieu.garel.free.fr/yolo/yolov4.h5)

Une fois le modèle récupéré, il vous faut créer votre version du fichier de config.

```
cp config_sample.json config_json
```

#### Description des champs de configuration
**ftp_server :** Nom du serveur FTP  
Si vous ne voulez pas utiliser de FTP, il faut laisser le champ vide ""  
Dans ce cas, la classification se fera via le répertoire local indiqué dans local_folder  
**ftp_username :** Username pour la connexion au serveur FTP  
**ftp_password :** Mot de passe pour la connexion au serveur FTP  
**ftp_directory :** Répertoire contenant les images sur le serveur FTP  
**local_folder :** En mode FTP, il s'agit du répertoire dans lequel les images seront téléchargées puis classifiées puis supprimées.   
En mode local, il s'agit du répertoire contenant les images à classifier  
**output_folder :** Répertoire dans lequel les fichiers de sortie seront stockés  
**model_file :** Chemin d'accès au fichier .h5 du modèle  
**treshold :** Valeur du seuil de classification pour le modèle. Cette valeur varie de 0 à 1. Plus la valeur est basse, plus nous sommes permissifs avec les classifications. Plus la valeur est haute, plus nous sommes restrictifs avec les classifications.  
**sequence_duration :** Valeur (en secondes) du temps de séquence. Le temps de séquence est utilisé par le script pour compter les groupes d'individus. Lors de la classification de l'image n, si l'image n-1 a été classifiée il y a moins du temps de séquence choisi alors le script considère qu'il s'agit du même groupe d'individus et donc il ne compte pas deux fois ce groupe.  
La valeur de base est de 10 secondes. Selon la fréquentation de votre sentier, vous pouvez baisser jusqu'à 5 s'il est très fréquenté et monter jusqu'à 15 s'il est très peu fréquenté. Au-delà de cet intervalle, les résultats sont moins bons.  
**time_step :** Pas de temps pour concaténer les classifications du modèle et sortir un fichier avec un nombre de passage en fonction du pas de temps choisi.  
Valeur de base : 'Hour', peut prendre les valeurs : 'Day', 'Month' et 'Year'  
**output_format :** Format du fichier de sortie.  
Valeur de base 'csv', peut prendre les valeurs : 'dat'  
**csv_column :** Modifie le mapping des colonnes dans le fichier de sortie  

Une fois le fichier de config modifié selon vos besoins. Vous pouvez créer un environnement virtuel python
```
python3 -m virtualenv venv
source venv/bin/activate
```
Ensuite, vous pouvez installer les lib nécessaire au fonctionnement du code via la commande :
```
pip install -e .
```

## Utilisation

Pour lancer le script, il faut taper :
```
python3 yolov4_attendance.py
```

N'oubliez pas de créer/modifier votre fichier de config !