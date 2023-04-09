# Ensemble-Techniques
Stacking:-

Stacking is a way of ensembling classification or regression models it consists of two-layer estimators. The first layer consists of all the baseline models that are used to predict the outputs on the test datasets. The second layer consists of Meta-Classifier or Regressor which takes all the predictions of baseline models as an input and generate new predictions.

Stacking Architecture:-

![image](https://user-images.githubusercontent.com/96175181/230754908-0d3fc68d-5501-4c3d-bb97-1fbce986ea13.png)

Why Stacking? 

Most of the Machine-Learning and Data science competitions are won by using Stacked models. They can improve the existing accuracy that is shown by individual models. We can get most of the Stacked models by choosing diverse algorithms in the first layer of architecture as different algorithms capture different trends in training data by combining both of the models can give better and accurate results.

What is a meta learner in machine learning?

Meta-learning helps researchers understand which algorithms generate the best/better predictions from datasets. Meta-learning algorithms use learning algorithm metadata as input. They then make predictions and provide information about the performance of these learning algorithms as output.
