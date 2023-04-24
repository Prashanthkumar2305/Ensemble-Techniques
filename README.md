# Ensemble-Techniques

1) Bagging
2) Boosting
3) Stacking


(i) Stacking:-

Stacking is a way of ensembling classification or regression models it consists of two-layer estimators. The first layer consists of all the baseline models that are used to predict the outputs on the test datasets. The second layer consists of Meta-Classifier or Regressor which takes all the predictions of baseline models as an input and generate new predictions.

Stacking Architecture:-

![image](https://user-images.githubusercontent.com/96175181/230754908-0d3fc68d-5501-4c3d-bb97-1fbce986ea13.png)

Why Stacking? 

Most of the Machine-Learning and Data science competitions are won by using Stacked models. They can improve the existing accuracy that is shown by individual models. We can get most of the Stacked models by choosing diverse algorithms in the first layer of architecture as different algorithms capture different trends in training data by combining both of the models can give better and accurate results.

What is a meta learner in machine learning?

-> In machine learning , a meta-classifier is essentially a meta learning algorithm that is used for classification predictive modeling tasks.

-> Meta - classifier is the classifier that makes a final prediction among all the predictions by using those predictions as features.



There are two ways to train meta-classifier:-

(1) Either you can use the output of the initial level classifier as inputs or features to your metaclassifier.

(2) Use probabilities of your first level classifier as features to your meta-classifier.



(ii) Bagging:-

> Bagging also known as bootstrap aggregation is the ensemble learning method that is commonly used to reduce variance within a noisy dataset.
> In bagging, a random sample of data in a training set is selected with replacement - meaning that the individual datapoints can be chosen more than once.


Bagging working:-

which has three basic steps:-

(1) Bootstrapping:- Bagging leverages a bootstrapping sampling technique to create diverse samples.This resampling method generates different subsets of the training dataset by selecting data points at random and with replacement.
This means that each time you select a data point from the training dataset, you are able to select the same instance multiple times. As a result a value / instance repeated twice(or more) in a sample.

(2) Parallel Training:-These bootstrap samples are then trained independently and in parallel with each other using weak or base learners.

(3) Aggregation:- Finally , depending on the task(i.e) regression or classification an average or a majority of the prediction are taken to compute a more accurate estimate.

-> In the case of regression, an average is taken of all the outputs predicted by the individual classifiers,this is known as soft voting.

-> For classification problems, the class with the highest majority of votes is accepted , this is known as hard voting or majority voting.


Benefits and challenges of bagging:-

(i) Ease of implementation

(2) Reduction of Variance

key challenges of bagging:-

(1) Loss of interpretability

(2) Computationally expensive

(3) Less flexible


