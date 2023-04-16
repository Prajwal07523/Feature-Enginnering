#!/usr/bin/env python
# coding: utf-8

# Q1: What are missing values in a dataset? Why is it essential to handle missing values? Name some 
# algorithms that are not affected by missing values.
# 
# Missing values in a dataset refer to the absence of a particular value for a specific variable in one or more observations in a dataset.
# 
# Handling missing values is essential because missing data can impact the accuracy of statistical analysis and modeling, as well as introduce biases in the results. Missing data can also reduce the sample size, which can decrease the statistical power of the analysis.
# 
# Some algorithms that are not affected by missing values include:
# 
# Decision tree,Random forest,K-nearest neighbors,Naive bayes

# Q2: List down techniques used to handle missing data.  Give an example of each with python code
# 
# There are several techniques used to handle missing data, some of which are:
# 
# Deletion: This involves deleting the missing values either from the entire dataset or from specific columns.
#     
# Imputation: This involves filling in the missing values with estimated values.
# 
# Interpolation: This involves estimating missing values by using the values of the adjacent data points.

# In[1]:


import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                   'B': [5, np.nan, 7, 8],
                   'C': [np.nan, 10, 11, 12]})


# In[3]:


df.dropna()


# In[6]:


df.fillna(df.mean())


# In[7]:


df.interpolate()


# Q3: Explain the imbalanced data. What will happen if imbalanced data is not handled?
# 
# Imbalanced data refers to a situation where the distribution of classes in a dataset is uneven, with one or more classes being significantly underrepresented compared to others.
# 
# If imbalanced data is not handled, it can lead to biased models that perform poorly on the underrepresented class(es). The model may simply predict the majority class for all instances, resulting in high accuracy but poor performance in terms of precision, recall, and F1-score, which are important metrics for evaluating classifier performance.

# Q4: What are Up-sampling and Down-sampling? Explain with an example when up-sampling and down-sampling are required.
# 
# Up-sampling and down-sampling are two resampling techniques used to address imbalanced data in classification problems.
# 
# Up-sampling: In up-sampling, the minority class is replicated multiple times to increase its representation in the dataset. This can be done randomly or systematically.
# 
# Down-sampling: In down-sampling, the majority class is randomly reduced in size to balance the representation of the minority class. This can result in loss of information, and should be used with caution.

# Q5: What is data Augmentation? Explain SMOTE.
# 
# Data augmentation is a technique used to artificially increase the size of a dataset by creating new synthetic samples that are similar to the existing ones.This method is useful when there is imbalanced dataset.
# 
# SMOTE (Synthetic Minority Over-sampling Technique) is a popular data augmentation technique used to address imbalanced data in classification problems. SMOTE creates synthetic samples of the minority class by interpolating between existing samples. It selects a sample from the minority class and computes the k nearest neighbors of this sample in the feature space. It then selects one of these neighbors at random and generates a new sample by linearly interpolating between the two samples. This process is repeated until the desired number of new samples is generated.

# Q6: What are outliers in a dataset? Why is it essential to handle outliers?
# 
# Outliers are observations in a dataset that significantly differ from other observations. They can be caused by measurement errors, data processing errors, or genuinely rare events. Outliers can affect the accuracy of statistical models by skewing the distribution of the data and influencing the estimates of central tendency and variability. Outliers can also affect the performance of machine learning models by introducing noise and bias in the data.
# 
# It is essential to handle outliers because they can have a significant impact on the analysis and results of a dataset

# Q7: You are working on a project that requires analyzing customer data. However, you notice that some of 
# the data is missing. What are some techniques you can use to handle the missing data in your analysis?
# 
# There are several techniques that can be used to handle missing data in a dataset. 
# Deletion: The rows and columns containing missing values are deleted.
# 
# Imputation: Imputation is a technique used to replace missing values with estimated values(ex: mean.median,mode).
# 
# Machine learning-based methods: Machine learning algorithms can be used to predict the missing values based on the available data. Techniques like Random Forest and XGBoost are popularly used for this purpose.

# Q8: You are working with a large dataset and find that a small percentage of the data is missing. What are 
# some strategies you can use to determine if the missing data is missing at random or if there is a pattern 
# to the missing data?
# 
# Visual inspection : inspect the missing data to determine if there is a pattern to the missing data using histograms or scattre plots.
# 
# Statistical tests: Statistical tests can be used to determine if there is a pattern to the missing data. ex: MCAR,MAR,MNAR.
# 
# Imputation and analysis: Another approach is to impute the missing values and analyze the data to see if the imputation affects the analysis.

# Q9: Suppose you are working on a medical diagnosis project and find that the majority of patients in the 
# dataset do not have the condition of interest, while a small percentage do. What are some strategies you 
# can use to evaluate the performance of your machine learning model on this imbalanced dataset?
# 
# 

# 10: When attempting to estimate customer satisfaction for a project, you discover that the dataset is 
# unbalanced, with the bulk of customers reporting being satisfied. What methods can you employ to 
# balance the dataset and down-sample the majority class?
# 
# Synthetic Minority Over-sampling Technique (SMOTE): In this method, new data points are synthesized for the minority class by interpolating between existing data points. This method can increase the size of the minority class while preserving the information and structure of the original dataset.

# Q11: You discover that the dataset is unbalanced with a low percentage of occurrences while working on a 
# project that requires you to estimate the occurrence of a rare event. What methods can you employ to 
# balance the dataset and up-sample the minority class?
# 
# Random over-sampling: In this method, new samples are randomly generated by duplicating the minority class data points. This method can be simple to implement but may result in overfitting to the minority class data points.
# 
# Synthetic Minority Over-sampling Technique (SMOTE): In this method, new data points are synthesized for the minority class by interpolating between existing data points. This method can increase the size of the minority class while preserving the information and structure of the original dataset.
# 
# Adaptive Synthetic Sampling (ADASYN): This method is similar to SMOTE, but instead of creating synthetic data points uniformly, it generates more synthetic data points in the vicinity of the minority class, making the classification boundary more balanced.

# In[ ]:




