# Statistical Learning, MESIO UPC-UB 

Chapter 1. Introduction.

## Introduction

## Pedro Delicado (UPC)

With minor changes made by Alex Sanchez (UB)

## 1. Machine learning and statistical learning

1. Machine learning and statistical learning

## Statistics, Machine Learning, Data Science

- Statistics is the science of collecting, analyzing, interpreting, and presenting data to extract meaningful insights, quantify uncertainty, and support decision-making under uncertainty (Efron and Hastie 2016).
- We define Machine Learning as a set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data (Murphy 2012).
- Statistical Learning refers to a framework for understanding and modeling complex datasets by using statistical methods to estimate relationships between variables, make predictions, and assess the reliability of conclusions. It bridges traditional statistics and machine learning by incorporating model interpretability, regularization techniques, and a focus on uncertainty quantification (Hastie, Tibshirani, and Friedman 2009)


## Statistics and Data Science (1)

Data Science: an interdisciplinary field combining techniques from statistics, machine learning, computer science, and domain-specific knowledge to extract insights and value from data. It encompasses the entire data lifecycle, including data collection, cleaning, exploration, modeling, interpretation, and communication of results to inform decision-making.

## Statistics and Data Science (2)

- Statistics provides the theoretical foundation for analyzing data, quantifying uncertainty, and drawing valid inferences. It ensures the rigor and interpretability of results.
- Machine Learning contributes computational and algorithmic tools that enable automated pattern detection and predictive modeling, often with a focus on scalability and performance.
- Statistical Learning serves as the bridge between statistics and machine learning, emphasizing interpretable models, regularization techniques, and uncertainty quantification.
- Data Science integrates all these components while also incorporating data engineering, visualization, and domain expertise to address real-world data-driven problems.
- Much of statistical technique was originally developed in an environment where data were scarce and difficult or expensive to collect, so statisticians focused on creating methods that would maximize the strength of inference one is able to make, given the least amount of data (Baumer, Kaplan, and Horton 2017).
- Much of the development of statistical theory was to find mathematical approximations for things that we couldn't yet compute (Baumer, Kaplan, and Horton 2017).
- Mathematics was the best computer (Efron and Hastie 2016).
- [FJrom the 1950s to the present is the "computer age" of [Statistics], the time when computation, the traditional bottleneck of statistical applications, became faster and easier by a factor of a million (Efron and Hastie 2016).

[^0]- The center of the field [Statistics] has in fact moved in the past sixty years, from its traditional home in mathematics and logic toward a more computational focus (Efron and Hastie 2016).
- Today, the manner in which we extract meaning from data is different in two ways, both due primarily to advances in computing:
- we are able to compute many more things than we could before, and;
- we have a lot more data than we had before.
(Baumer, Kaplan, and Horton 2017)
- Michael Jordan (UC Berkeley) has described Data Science as the marriage of computational thinking and inferential thinking. [...] These styles of thinking support each other.
- Data Science is a science, a rigorous discipline combining elements of Statistics and Computer Science, with roots in Mathematics.

From Efron and Hastie (2016):

- A particularly energetic brand of the statistical enterprise has flourished in the new century, data science, emphasizing algorithmic thinking rather than its inferential justification.
- Data Science [...] seems to represent a statistics discipline without parametric probability models or formal inference.
- [Machine Learning] large-scale prediction algorithms (neural nets, deep learning, boosting, random forests, and support-vector machines) are the media stars of the big-data era.
- Why have they taken center stage?
- Prediction is commercially valuable.
- Prediction is the simplest use of regression theory.
- It can be carried out successfully without probability models, perhaps with the assistance of cross-validation, permutations, bootstrap.

Different focus of Statistics and Machine Learning:

|  | Asymptotics, <br> optimality | Interpretability | Accurate |  |
| :--- | :---: | :---: | :---: | :---: |
| prediction | Scalability |  |  |  |
| Statistics | $X X X X X$ | $X X X X X$ | $X X$ | $X$ |
| Machine | $X$ | $X X$ | $X X X X X$ | $X X X X X$ |
| Learning | $X$ |  |  |  |

## Statistical Learning, Machine Learning (I)

- This course is about learning from data, and specifically focused on the prediction problem.
- We are studying models, tools and techniques originally developed by statisticians (sparse estimation of linear regression models and generalized linear models, nonparametric versions of them, classification and regression trees, ...)
- Nowadays they are part of the toolkit of Machine Learning practitioners (usually coming from Computer Science) and Data Scientists in general.
- We also will learn algorithms and procedures that has been proposed by researchers in Machine Learning (neural networks, support vector machines, boosting, random forests, ...) that now are part of the statisticians' prediction toolkit.
- Since computation plays a key role [in learning from data], it is not surprising that much of the new development has been done by researchers in other fields such as computer science and engineering (Hastie, Tibshirani, and Friedman 2009).


## Statistical Learning, Machine Learning (II)

- Which would have been the right name for this course?
- Learning from data.
- Machine Learning.
- Statistical Learning.
- We the instructors are statisticians. So this course will necessarily be different from a Machine Learning course in a Computer Science MSc program, even if the contents of both could have a large intersection.
- We feel comfortable with Statistical Learning, a term that started to be popular from the publication in 2001 of the first edition of The Elements of Statistical Learning, with a second edition in 2009 (Hastie, Tibshirani, and Friedman 2009).
- See also James, Witten, Hastie, and Tibshirani (2013) and Hastie, Tibshirani, and Wainwright (2015).

Baumer, B. S., D. T. Kaplan, and N. J. Horton (2017).
Modern Data Science with $R$.
CRC Press.
Efron, B. and T. Hastie (2016).
Computer Age Statistical Inference.
Cambridge University Press.
Hastie, T., R. Tibshirani, and J. Friedman (2009).
The Elements of Statistical Learning (2nd ed.).
Springer.
Hastie, T., R. Tibshirani, and M. Wainwright (2015).
Statistical learning with sparsity: the lasso and generalizations.
CRC Press.
James, G., D. Witten, T. Hastie, and R. Tibshirani (2013).
An Introduction to Statistical Learning with Applications in $R$.
Springer.
Murphy, K. P. (2012).
Machine Learning: A Probabilistic Perspective.
The MIT Press.


[^0]:    ${ }^{1}$ Slides with colored background are optional material.

