# Base PyTorch ADR Pipeline

Cette branche constitue la branche de référence PyTorch du projet. Elle présente la version canonique du modèle PI-DeepONet appliqué à l’équation d’advection-diffusion-réaction paramétrique, sans la couche complète d’organisation dédiée à la comparaison JAX contre PyTorch.

## Introduction

Le problème physique étudié dans cette branche est une équation d’advection-diffusion-réaction unidimensionnelle de la forme

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3).
\]

La quantité \(u(x,t)\) dépend de la position \(x\) et du temps \(t\). Les différents termes de l’équation ont une interprétation physique simple.

- \(u_t\) décrit l’évolution temporelle de la solution.
- \(v\,u_x\) représente l’advection, c’est-à-dire le transport de la quantité étudiée.
- \(D\,u_{xx}\) représente la diffusion, qui tend à lisser spatialement la solution.
- \(\mu (u-u^3)\) représente un terme de réaction non linéaire cubique.

Ce type d’équation intervient dans des problèmes où une grandeur est à la fois transportée, diffusée et transformée localement. Dans ce projet, l’objectif n’est pas de résoudre un cas unique, mais de construire un surrogate neural capable de généraliser sur une famille de problèmes ADR paramétriques.

## Présentation du Squelette de la Branche

Cette branche est organisée comme une branche baseline-only. Elle ne cherche pas à porter toute la couche de comparaison inter-frameworks. Sa logique est de rendre lisible et reproductible le pipeline PyTorch de référence.

- `code/`: code source, configurations, scripts de lancement et tests
- `figures/`: figures de référence du solveur classique, du PI-DeepONet seul, et des comparaisons PI-DeepONet vs Crank-Nicolson
- `assets/`: visuels les plus représentatifs pour une lecture rapide
- `models/`: poids sauvegardés de référence
- `README.md`: présentation scientifique complète de la branche
- `Makefile`: commandes utilitaires usuelles
- `requirements-base.txt`: environnement Python de référence pour cette branche

Le lecteur qui arrive sur cette branche doit pouvoir comprendre rapidement :

1. quel problème physique est traité ;
2. quelle architecture est utilisée ;
3. comment l’entraînement est organisé ;
4. quels résultats doivent être considérés comme les résultats PyTorch de référence.

## Formulation du Problème ADR Paramétrique

Le problème étudié est paramétrique. Le réseau n’apprend donc pas une solution unique, mais un opérateur capable de prédire la solution pour différentes combinaisons de paramètres physiques et de conditions initiales.

Les coefficients physiques varient dans les intervalles suivants :

- \(v \in [0.5, 1.0]\)
- \(D \in [0.01, 0.2]\)
- \(\mu \in [0.0, 1.0]\)

Les conditions initiales sont également paramétrées :

- \(A \in [0.7, 1.0]\)
- \(\sigma \in [0.4, 0.8]\)
- \(k \in [1.0, 3.0]\)
- \(x_0 = 0\)

Le domaine spatial est

\[
x \in [-5, 8],
\]

et l’horizon temporel de la baseline complète est

\[
T_{\max} = 3.0.
\]

La discrétisation de référence utilisée pour les audits de la baseline est :

- \(N_x = 500\)
- \(N_t = 200\)

Trois familles principales de conditions initiales sont considérées :

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

Cette diversité de familles est essentielle. Elle permet d’évaluer la capacité du modèle à généraliser sur des régimes qualitativement différents, plutôt que sur une seule forme initiale.

## Solveur Numérique de Référence : Crank-Nicolson

Les prédictions du réseau sont évaluées contre un solveur numérique de référence de type Crank-Nicolson.

Crank-Nicolson est un schéma implicite classique de discrétisation temporelle pour les équations aux dérivées partielles dépendant du temps. Il présente un bon compromis entre stabilité numérique et précision, ce qui en fait une référence pertinente pour évaluer un surrogate neural.

Dans cette branche, le solveur Crank-Nicolson joue deux rôles :

- fournir la vérité terrain numérique contre laquelle les prédictions du réseau sont comparées ;
- servir de baseline temporelle pour mesurer le gain d’inférence du PI-DeepONet.

L’intérêt scientifique de ce choix est important : la qualité du réseau n’est pas jugée uniquement à partir de sa loss d’entraînement, mais par rapport à une solution numérique de référence issue d’une méthode bien établie.

## Description du Réseau de Neurones Utilisé

Le modèle utilisé dans cette branche est un PI-DeepONet, c’est-à-dire un Deep Operator Network entraîné avec une contrainte physique explicite via le résidu de l’équation ADR.

L’architecture repose sur une séparation branch/trunk.

- La branche `branch` encode les paramètres physiques et les paramètres décrivant la condition initiale.
- La branche `trunk` encode les coordonnées du point d’évaluation \((x,t)\).
- Les deux représentations sont fusionnées par des transformations conditionnelles de type FiLM.

Les dimensions d’entrée sont :

- branche : 8 variables \((v, D, \mu, type, A, x_0, \sigma, k)\)
- trunk : 2 variables \((x,t)\)

L’architecture de référence est :

- profondeur branch : 5
- profondeur trunk : 4
- largeur branch : 256
- largeur trunk : 256
- dimension latente : 256
- nombre de Fourier features : 20
- échelles de Fourier : \(0, 1, 2, 3, 4, 5, 6, 8, 10, 12\)
- activation : SiLU

Le trunk utilise un encodage de Fourier multiscale afin d’améliorer la représentation de structures oscillantes ou localisées, en particulier pour les familles `Sin-Gauss` et `Gaussian`.

La fonction de coût combine trois termes :

- perte de résidu PDE ;
- perte sur la condition initiale ;
- perte sur les conditions aux bords.

L’optimisation principale est réalisée avec Adam. La branche comporte également un mécanisme de finisher L-BFGS dans certaines phases de polissage. La learning rate de référence est d’environ \(6.08 \times 10^{-5}\), avec une décroissance explicite au cours des phases longues d’entraînement.

Les hyperparamètres principaux de la baseline sont :

- batch size : 8192
- nombre de points échantillonnés : 12288
- warmup : 7000 itérations
- itérations par fenêtre temporelle : 8000
- itérations de correction : 9000
- nombre de boucles externes : 3
- rolling window : 2000
- nombre maximal de retries : 4

## Protocole Expérimental du Modèle de Base

L’objectif de l’entraînement de la baseline est de construire un surrogate PyTorch fiable sur l’ensemble du problème ADR paramétrique, et non seulement sur un cas simple ou une famille isolée.

La stratégie d’entraînement est progressive. Elle repose sur un curriculum temporel et sur plusieurs mécanismes de contrôle visant à stabiliser l’apprentissage.

Les mécanismes principaux sont :

- `warmup` initial sur la condition initiale ;
- entraînement par fenêtres temporelles successives ;
- stratégie `king of the hill` pour conserver le meilleur état du modèle ;
- `rollback` et retry lorsqu’une fenêtre ne satisfait pas les critères de validation ;
- ajustement adaptatif du poids PDE par heuristique inspirée du NTK ;
- correction ciblée sur les familles difficiles détectées lors des audits ;
- phase finale de polissage plus stricte.

Le curriculum temporel de la baseline est défini par trois zones :

- de \(t=0\) à \(t=0.05\) : pas de temps \(0.01\)
- de \(t=0.05\) à \(t=0.30\) : pas de temps \(0.05\)
- de \(t=0.30\) à \(t=3.0\) : pas de temps \(0.10\)

Les poids de loss de départ et d’arrivée jouent un rôle important :

- poids initial de la condition initiale : 2000
- poids final de la condition initiale : 100
- poids des bords : 200
- poids PDE initial : 500

Les critères de validation utilisés dans les audits sont :

- seuil condition initiale : 0.02
- seuil étape temporelle : 0.03

L’idée centrale de ce protocole est que la convergence est définie par la qualité effective de la solution produite, et pas seulement par la baisse d’une loss globale. Cette logique fait de la branche `base` une vraie baseline scientifique, et pas seulement un script d’entraînement.

## Résultats Numériques du Modèle de Base

La conclusion principale de cette branche est positive : le PI-DeepONet PyTorch apprend un surrogate de bonne qualité pour le problème ADR multifamille.

Sur le benchmark multifamille de référence avec 20 cas d’évaluation par famille, les résultats obtenus sont :

- erreur relative \(L^2\) globale : `0.00507 ± 0.00392`
- `Tanh` : `0.00139 ± 0.00035`
- `Sin-Gauss` : `0.00978 ± 0.00286`
- `Gaussian` : `0.00405 ± 0.00100`

Les mesures d’inférence donnent :

- temps d’inférence full grid : `0.210 s`
- temps d’inférence en saut temporel : `0.00285 s`
- temps du solveur Crank-Nicolson de référence : `0.499 s`
- speedup en saut temporel : `×175.03`

Le temps total d’entraînement du benchmark multifamille court correspondant est :

- temps total : `5329.21 s`

Interprétation scientifique :

- la précision globale est suffisamment faible pour faire du modèle un surrogate crédible ;
- le modèle généralise bien sur plusieurs familles de conditions initiales ;
- la famille `Sin-Gauss` reste la plus difficile ;
- le gain d’inférence justifie l’intérêt pratique du surrogate.

Cette section constitue le résultat principal de la branche `base`.

## Résultats Numériques JAX

Cette branche n’est pas la branche de comparaison, mais il reste utile de situer la baseline PyTorch par rapport aux résultats JAX obtenus sous protocole apparié.

Sur le benchmark multifamille strict de comparaison, les résultats JAX observés sont :

- erreur relative \(L^2\) globale : `1.66884 ± 1.62812`
- `Tanh` : `1.23642 ± 0.15997`
- `Sin-Gauss` : `2.63937 ± 2.54170`
- `Gaussian` : `1.13073 ± 0.21905`

Les mesures de temps correspondantes sont :

- temps d’inférence full grid : `0.249 s`
- temps d’inférence en saut temporel : `0.01079 s`
- speedup vs Crank-Nicolson : `×45.38`
- temps total d’entraînement : `349.13 s`

L’analyse par famille et par ablation montre globalement que :

- JAX entraîne plus vite ;
- PyTorch produit des solutions nettement plus fiables dans le protocole de référence ;
- les familles `Sin-Gauss` et `Gaussian` concentrent l’essentiel de la difficulté ;
- l’introduction d’un ansatz sur la condition initiale améliore fortement les cas les plus difficiles, davantage qu’un simple polissage L-BFGS.

Cette section est donnée ici comme élément de contexte, mais la discussion détaillée de ces résultats relève de la branche `jax-comparison`.

## Bilan

La branche `base` permet d’établir trois conclusions principales.

Premièrement, un PI-DeepONet PyTorch correctement structuré permet d’apprendre un surrogate précis pour l’équation ADR paramétrique. Deuxièmement, cette précision s’accompagne d’un gain d’inférence significatif par rapport au solveur Crank-Nicolson de référence. Troisièmement, la baseline PyTorch constitue la base scientifique solide du projet, sur laquelle viennent ensuite se greffer les comparaisons méthodologiques et les ablations.

Autrement dit, cette branche répond à la question fondamentale : est-ce que l’approche fonctionne réellement sur le problème cible ? La réponse est oui.

## Logique d’Utilisation de la Branche

L’environnement de cette branche est volontairement limité à la baseline PyTorch.

Installation recommandée :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
```

Commandes utiles :

Installer les dépendances :

```bash
make install
```

Lancer les tests :

```bash
make test
```

Vérifier la compilation du code :

```bash
make check
```

Lancer l’entraînement principal :

```bash
make train
```

Lancer l’analyse globale PI-DeepONet vs Crank-Nicolson :

```bash
make analysis
```

Lancer le benchmark d’inférence :

```bash
make benchmark
```

Points d’entrée directs :

- entraînement : `python code/scripts/train.py`
- tuning : `python code/scripts/tune_optuna.py`
- tests : `python -m pytest -q code/tests`

Navigation recommandée :

1. lire ce `README.md` ;
2. regarder `code/configs/` ;
3. lire `code/scripts/` et `code/src/training/` ;
4. consulter `figures/`, `assets/` et `models/`.
