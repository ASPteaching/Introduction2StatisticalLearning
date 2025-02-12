# Statistical Learning

# Introduction 

This github repository is intended to provide materials 
(slides, scripts datasets etc) for part of the _Statistical Learning_ 
course at the [UPC-UB MSc in Statistics and Operations Research (MESIO)](https://mesioupcub.masters.upc.edu/en).

This part of the course has an introduction and two blocks, each with two parts.

0. Introduction

1. Tree based methods

    1.1 Decision trees

    1.2 Ensemble methods

2.  Artificial neural networks

    2.1 Artificial neural networks

    2.2 Introduction to deep learning

# Class material

All class materials are available from the repository [https://aspteaching.github.io/Introduction2StatisticalLearning/](https://aspteaching.github.io/Introduction2StatisticalLearning/).

In this page you will find links to the html/pdf version of the slides and other documents, as well as to datasets or references and resources documents

- [Course presentation](https://github.com/ASPteaching/Introduction2StatisticalLearning/blob/main/0-Course_presentation_and_Introduction/Course_Presentation-SL.pdf)

## Introduction to Statistical Learning

- [Introduction: Statistics, Machine Learning, Statistical Learning and Data Science](https://github.com/ASPteaching/Introduction2StatisticalLearning/blob/main/0-Course_presentation_and_Introduction/C0-Intro2StatLearn-PD-AS.pdf)
- [Overview of Supervised Learning](https://github.com/ASPteaching/Introduction2StatisticalLearning/blob/main/0-Course_presentation_and_Introduction/C1-SupervisedLearning-PD_AS.pdf)
- [Rlabs](https://github.com/ASPteaching/Introduction2StatisticalLearning/tree/main/labs/intro2StatLearn)
  - Regression with KNN
  - Classification with KNN

- Complements
  - [Introduction to biomarkers and diagnostic tests](https://github.com/ASPteaching/Introduction2StatisticalLearning/blob/main/0-Course_presentation_and_Introduction/From%20Biomarker%20to%20Diagnostic%20Tests.pdf)
  - [Building and validating biomarkers](https://github.com/uebvhir/Pindoles/raw/master/2019_02_28_Busqueu_fama_Estrategies_modelitzacio.pdf)

## Decision Trees

Decision trees are a type of non-parametric classifiers which have been Very successful because of their interpretability, flexibility and a very decent accuracy.

-   [Slides](https://aspteaching.github.io/Introduction2StatisticalLearning/1.1-DecisionTrees-Slides.html)
-   [Notes](https://aspteaching.github.io/Introduction2StatisticalLearning/1.1-DecisionTrees.html)
-   [R-lab](https://aspteaching.github.io/Introduction2StatisticalLearning/labs/DecisionTrees/CART-Examples.html)
-   Python-labs
    -   [Introduction to python (from ISL. Ch 02)](https://aspteaching.github.io/Introduction2StatisticalLearning/labs/Ch02-statlearn-lab.ipynb)
    -   [Decision Trees lab (from ISL. Ch 08)](https://aspteaching.github.io/Introduction2StatisticalLearning/labs/DecisionTrees/ISLch08-baggboost-lab.ipynb)

## Ensemble methods

The term "Ensemble" (together in french) refers to distinct approaches to build predictiors by combining multiple models.

They have proved to addres well some limitations of trees therefore improving accuracy and robustness as well as being able to reduce overfitting and capture complex relationships.

## Artifical Neural Networks

Thesea are raditional ML models, inspired in brain, that simulate neuron behavior, thata is they receive an input, which is processed and an  output prediction is produced.

For long their applicability has been relatively restricted to a few fields or problems due mainly to their "black box" functioning that made them hard to interpret.

The scenario has now completely changed with the advent of deep neural networks which are in the basis of many powerful applications of artificial intelligence.

## Deep learning

Esssentially these are ANN with multiple hidden layers with allow overpassing many of their limitations.
They can be tuned in a much more automatical way and have been applied to many complex tasks. such as Computer vision, Natural Language Processing or Recommender systems.


# References and resources

## References for Tree based methods

- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.

- Brandon M. Greenwell (202) Tree-Based Methods for Statistical Learning in R. 1st Edition. Chapman and Hall/CRC DOI: https://doi.org/10.1201/9781003089032 Web site

- Efron, B., Hastie T. (2016) Computer Age Statistical Inference. Cambridge University Press. Web site

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction. Springer.

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning (Vol. 112). Springer.

## References for deep neural networks

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning (Vol. 1). MIT press. Web site

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

- Chollet, F. (2018). Deep learning with Python. Manning Publications.

- Chollet, F. (2023). Deep learning with R . 2nd edition. Manning Publications.

## Some interesting online resources

-[Decision Trees free course (9 videos). By Analytics Vidhya](https://www.youtube.com/playlist?list=PLdKd-j64gDcC5TCZEqODMZtAotCfm5Zkh)

- [Applied Data Mining and Statistical Learning (Penn Statte-University)](https://online.stat.psu.edu/stat508/)

- [R for statistical learning](https://daviddalpiaz.github.io/r4sl/)

- [An Introduction to Recursive Partitioning Using the RPART Routines](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf)

- [Introduction to Artificial Neural Networks](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf)
