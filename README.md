# Diagnostic multimodal par IA : Cancers sanguins & Tumeurs cérébrales via Transfer Learning

## Contexte du projet
Ce projet a pour objectif de concevoir une **solution unifiée d’analyse d’images médicales** basée sur le deep learning, combinant :  

- La **détection d’objets** pour les tumeurs cérébrales à partir d’IRM ou de scanners cérébraux.  
- La **classification fine de cellules sanguines anormales** à partir de frottis sanguins.  

L’objectif est d’automatiser l’analyse de ces pathologies critiques afin d’améliorer la précision du diagnostic et d’accélérer le traitement des patients.

## Partie 1 — Classification des cellules sanguines cancéreuses (PyTorch)

### Tâches réalisées
- Import des bibliothèques nécessaires (`torch`, `torchvision`, `numpy`, `matplotlib`, etc.)  
- Chargement et vérification des images du dataset (formats acceptés : jpeg, jpg, bmp, png) avec suppression des fichiers non conformes et gestion des erreurs via `try-except`.  
- Exploration des classes du dataset (les noms des dossiers représentent les classes) et visualisation du nombre d’échantillons par classe via `countplot`.  
- Affichage d’un échantillon d’images pour chaque classe.  
- Division des images en trois ensembles :  
  - **Train** : 70 %  
  - **Validation** : 15 %  
  - **Test** : 15 %  
- Application de transformations sur les données d’entraînement pour équilibrer les classes et augmenter le nombre d’images : `blur`, `noise`, `flip`.  
- Utilisation des `Transforms` de PyTorch pour redimensionner, convertir en tenseurs et normaliser les images.  
- Création de `DataLoaders` pour le chargement par batch et le shuffling des données.  
- Utilisation d’un modèle pré-entraîné **GoogLeNet**, avec remplacement de la couche fully connected (FC) par un réseau adapté à la classification.  
- Détermination du **learning rate**, de la **loss function** et de l’**optimizer** appropriés.  
- Évaluation du modèle sur les ensembles de validation et test pour mesurer précision et capacité à généraliser.  
- Sauvegarde du modèle entraîné.  

---

## Partie 2 — Détection des tumeurs cérébrales avec YOLOv8

### Tâches réalisées
- Affichage d’un échantillon d’images avec les **boîtes englobantes** pour chaque classe.  
- Filtrage des images et labels :  
  - Vérification de la présence du fichier `.txt` correspondant à chaque image.  
  - Copie des images et labels valides dans `outputpath/images/(train|valid|test)` et `outputpath/labels/(train|valid|test)`.  
  - Affichage d’un message pour les images sans labels et exclusion de celles-ci.  
- Création de fichiers de configuration `data.yaml` (sans augmentation) et `data2.yaml` (avec augmentation).  
- Comptage des images et labels dans les ensembles d’entraînement et validation.  
- Vérification de la correspondance entre chaque image et son label : suppression des éléments sans correspondance.  
- Entraînement du modèle YOLOv8 avec les hyperparamètres appropriés.  
- Évaluation des performances, précision et capacité de généralisation sur données inédites.  
- Sauvegarde du modèle entraîné.  

---

## Partie 3 — Interface utilisateur

- Développement d’une **interface interactive avec Streamlit** permettant :  
  - L’upload d’images IRM ou frottis sanguins.  
  - La classification automatique des cellules sanguines cancéreuses.  
  - La détection et localisation des tumeurs cérébrales.  
  - La visualisation des résultats directement dans l’interface.  

---

## Technologies et bibliothèques utilisées
- **Python 3.10+**  
- **PyTorch** (pour CNN et GoogLeNet)  
- **torchvision** (transforms, datasets)  
- **YOLOv8** (détection d’objets)  
- **OpenCV & PIL** (prétraitement des images)  
- **Matplotlib / Seaborn** (visualisation)  
- **Streamlit** (interface interactive)  
- **Numpy & Pandas** (manipulation de données)  
