#!/usr/bin/env python
# coding: utf-8

# ## Real Estate Price Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#data frame name housing
housing=pd.read_csv("data.csv")
housing.head()


# In[3]:


housing.info()
#help in checking missing data


# In[4]:


housing["CHAS"].value_counts()
#0 in how many entries and 1 in how many entries


# In[5]:


housing.describe()
#from count we know the number of row shaving null values


# In[6]:


housing.tail()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
#to visualise graph here only


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


# housing.hist(bins=50,figsize=(20,15))
# #dont need plt.show()in jupyter


#  ## Train-Test Splitting

# In[10]:


# import numpy as np 

# #np.random.seed(42) means as many times u run it will remain same
# #dont use random bcz test data will showed in training data when ever executing aprogram
# #to avoid overfitting model will memorise all data
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


# train_set,test_set=split_train_test(housing,0.2)


# In[12]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(f"Rows in test set: {len(test_set)} Rows in train set: {len(train_set)}")


# In[13]:


#If CHAS is an important feature if it is on equal part train and test data shouold represent total population
#so use stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["CHAS"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[14]:


strat_train_set.info()


# In[15]:


strat_test_set["CHAS"].value_counts()


# In[16]:


strat_train_set["CHAS"].value_counts()


# In[17]:


housing=strat_train_set.copy()


# ## Looking For Correlations

# In[18]:


#inbuilt function in pandas to generate corelation matrix
corr_matrix=housing.corr()


# In[19]:


#see pearson corelation matrix 1 refers to high corelation lie betn -1 to +1 seee +ve and  -ve corelation
corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


# from pandas.plotting import scatter_matrix
# #plot only strong relation values
# attributes=["MEDV","RA","ZN","LSTAT"]
# scatter_matrix(housing[attributes],figsize=(12,8))


# In[21]:


# housing.plot(kind="scatter",x="RA",y="MEDV",alpha=0.8)


# In[22]:


#we can remove the outliers
housing["TAXRA"]=housing["TAX"]/housing["RA"]
housing.head()


# In[23]:


# attributes=["MEDV","RA","ZN","LSTAT"]
# scatter_matrix(housing[attributes],figsize=(12,8))


# In[24]:


# housing.plot(kind="scatter",x="TAX",y="MEDV",alpha=0.8)


# In[25]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# # #Taking Care Of Missing DATA

# In[26]:


a=housing.dropna(subset=["RA"])
a.shape
# here housing data frame is not changed to change it pass inplace=True inside bracket


# In[27]:


#option2
housing.drop("RA",axis=1).shape
# there will be no RA coulmn shown but originally is not changed


# In[28]:


# option 3 replace with median mean or mode
median=housing["RA"].median()
housing["RA"].fillna(median)


# In[29]:


housing.shape


# In[30]:


housing.describe()
# before starting imputer


# In[31]:


#if test set have missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median");
imputer.fit(housing)
# fit imputer to housing data


# In[32]:


imputer.statistics_
# reeplace every null value with median in every attribute


# In[33]:


X=imputer.transform(housing)


# In[34]:


housing_tr=pd.DataFrame(X,columns=housing.columns)
#transformed missing value of missing data


# In[35]:


housing_tr.describe()


# ## SK learn Library 

# In[36]:


#sk learn has 3 objects
# 1 Estimators-It estimates some parameter based on a data set like imputer
# it has a fit method and transform method
#Fit-method-Fits The Data SEt and Calculates Internal Parameter has hyper parameter like strategy
#Transform Method-Take input and returns outp
#Tra


# In[37]:


# building pipeline and before it automate the data set


# ## Feature Scaling

# In[38]:


# primarily 2 types of feature scaling Method
# 1.Min-max Scaling(Normalisation)
# (value-min)/(max-min)
# here value lie in certain range ie from 0 to 1
# Sklearn provides MinMAxScaler for this
# 2.Standardisation
#  (value-mean)/standarddeviation
#  Sklearn provides StandardScaler for this



# In[39]:


from sklearn.pipeline import Pipeline
# feature scaling scale feature value to cer     
from sklearn.preprocessing import StandardScaler
# add as many things in pipeline ,pipeline takes a string
my_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),
                     ("std_scaler",StandardScaler())])


# In[40]:


housing_num_tr=my_pipeline.fit_transform(housing)
#every process u done must be in pipeline so transform before imputed data to the pipe line 


# In[41]:


housing_num_tr
#it is an numpy array predictor use numpy array as input


# ## Selecting A Desired Model

# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[43]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[44]:


list(some_labels)


# ## Evaluating The Model

# In[45]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[46]:


rmse
# it has 0 errror bcz it deeply understand the model and got over fitted also learned the noise 


# ## Using Better Evaluation Technique And Cross Validation

# In[47]:


#1 2 3 4  5  67 8 9 
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
#do or 10 folds our scores here negative
rmse_scores=np.sqrt(-scores)
rmse_scores
# for prices to have 24 32 36 etc 4 to 5 error is ok


# In[48]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("standard Deviation:", scores.std())


# In[49]:


print_scores(rmse_scores)


# ## Saving The Model

# In[52]:


from joblib import dump, load
dump(model, "RealEstate.joblib")


# ## TESTING THE MODEL

# In[55]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
#print(final_predictions,list(Y_test))


# In[56]:


final_rmse


# In[57]:


prepared_data[0]


# In[ ]:




