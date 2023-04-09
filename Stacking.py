#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv(r"C:\Users\Divyashree K\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\dataset.csv")


# In[3]:


display(df.dtypes)


# In[4]:


df


# In[5]:


# Creating X and Y for training
X=df.drop(['target'],axis=1)
Y=df['target']
x1=X.columns
x1


# In[6]:


# standardization
#The goal of standardization is to enforce a level of consistency or uniformity to certain practices or operations within the selected environment

sc=StandardScaler()
X=sc.fit_transform(X)
X=pd.DataFrame(X,columns=x1)


# In[7]:


X.head()


# In[8]:


Y.head()


# In[9]:


# split the dataset into train and test datasets about 20% of data.
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[10]:


x_train


# ## Building 1st layer Estimators

# In[11]:


knc=KNeighborsClassifier()
nb=GaussianNB()


# ### Let’s Train and evaluate with our first layer estimators to observe the difference in the performance of the stacked model and general model
# 

# ## Model training
# ### KNeighborsClassifier

# In[12]:


model_knc=knc.fit(x_train,y_train)
pred_knc=model_knc.predict(x_test)


# ### Evaluation for KNeighborsClassifier

# In[13]:


acc=accuracy_score(y_test,pred_knc)
print("Accuracy of KNeighnorsclassifier algorithms is:",round(acc*100,2))


# ### Naive_Bayes

# In[14]:


model_nb=nb.fit(x_train,y_train)
pred_nb=model_nb.predict(x_test)


# ### Evaluation for Naive_bayes

# In[15]:


acc2=accuracy_score(y_test,pred_nb)
print("Accuracy of NaiveBayes algorithms is:",round(acc2*100,2))


# # Implimenting the Stacking

# In[16]:


lr=LogisticRegression()  # Defining meta classifier
clf_stack=StackingClassifier(classifiers=[knc,nb],meta_classifier=lr,use_probas=True,use_features_in_secondary=True)

> use_probas=True indicates the Stacking Classifier uses the prediction probabilities as an input instead of using predictions    classes.
> use_features_in_secondary=True indicates Stacking Classifier not only take predictions as an input but also uses features in   the dataset to predict on new data. 
# ## Training  stack 

# In[17]:


model_stack=clf_stack.fit(x_train,y_train)
pred_stack=model_stack.predict(x_test)


# In[18]:


acc3=accuracy_score(y_test,pred_stack)
print("Accuarcy of stacking algoritms is:",round(acc3*100,2))


# #### Our both individual models scores an accuracy of nearly 91.8% as well as 86.89 and our Stacked model got an accuracy of nearly 90.16. By Combining two individual models we got a significant performance improvement
# 

# In[ ]:




