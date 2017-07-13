# Atelier « Deep Learning » 01 :

Objectif de l'atelier :
Savoir utiliser les fonctionnalités de base de la librairie Python Keras.
La tâche est une tâche de reconnaissance d'entités nommées dans un corpus.

Prérequis :
- Maîtriser le langage Python
- Avoir pu installer Keras (avec backend Theanos) -> une machine virtuelle VirtualBox avec l'installation circule 

Description des fichiers :
- ScriptsTest.py : script contenant plusieurs fonctions utilitaires ainsi qu'un exemple d'utilisation fonctionnelle de Keras
- test.txt : Contient environ 23 000 tokens labélisés (fichier à n'utiliser que pour l'évaluation) 
- test_withoutLabels.txt : Contient environ 23 000 tokens à labeliser (fichier à utiliser pour la prédiction)
- train.txt : Contient environ 130 000 tokens labelisés (fichier à utiliser pour l'entraînement) 
- word2vecData_embeddings_dim100.txt : Contient les vecteurs associés à chaque mots des corpus train+test (dimension 100).
- word2vecData_embeddings_dim200.txt : Contient les vecteurs associés à chaque mots des corpus train+test (dimension 200).
Remarque : Ne contient pas de mots-outils, ni de caractères. Vous pouvez donc trouver dans les train/test des tokens sans équivalent dans ce fichier.
(- word2vecData_embeddings_dim200.gensim : Contient l'objet "model" tel que Gensimn le structure - à n'utiliser que si vous savez utiliser Gensim)

Formats :
1) Pour test.txt : Un token par ligne
2) Pour train.txt et test_withoutLabels.txt : token	label (séparé par une tabulation)
3) Pour word2vecData_embeddings_dim200.txt : token valeur1 valeur2 ... valeurN (séparé par un espace)

Informations sur les embeddings :
Collected 10083 word types from a corpus of 82163 raw words and 6339 sentences.

Objectif de la tâche :
Utiliser en entrée les embeddings des mots pour entraîner un réseau de neurones à reconnaître les mots représentant tout ou partie d'un terme désignant une maladie.
On pourra simplement utiliser en sortie une seule valeur : 1 si c'est une maladie, 0 sinon.

Remarques :
- Il y a quelques données mal fromatées dans les batch train/test...

"Corrections" de l'atelier :
Le script Python de ce répertoire montre un exemple complet de programme pour répondre à la tâche en utilisant la librairie Keras. Je pense que le réseau implémenté est celui-ci :
![alt text](https://github.com/ArnaudFerre/AtelierDeepLearningILES/blob/master/workshops/workshop_01/Task_01/pictures/DL01.png)

Néanmoins, je n'en suis pas si sûr, et ça pourrait peut-être être un de ceux-ci :
![alt text](https://github.com/ArnaudFerre/AtelierDeepLearningILES/blob/master/workshops/workshop_01/Task_01/pictures/DL02.png)

Si vous exécutez ce script sans changer ses paramètres, vous devriez avoir des résultats autour de ceux-là (les résultats sont variables !) :
- précision : 0.74 (j'ai pu monter à 0.78 parfois)
- rappel : 0.35
- F-mesure : 0.48
