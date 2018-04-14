
# coding: utf-8

# Combining the best features that we got from boruta, rfe and based on importance of features of different models

# In[89]:


import pandas as pd
import datetime
import numpy as np
import pickle
import sklearn
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[90]:


col_list=['Page_popularity','Page_visited_no_of_times','Page_talking_about','Page_category','c1','c2','c3','c4','c5','c6','c7','c8',
         'c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','CC1','CC2','CC3','CC4','CC5',
         'Base_time','Post_length_char_count','Post_share_count','Post_promoted','Time_target','Sunday_post','Monday_post',
         'Tuesday_post','Wednesday_post','Thrusday_post','Friday_post','Saturday_post','Sunday_base','Monday_base','Tuesday_base','Wednesday_base',
         'Thrusday_base','Friday_base','Saturday_base','Target_variable']
##############################################################################################################################################################
#write code to fetch dataset file from s3 ,unzip it and load it into dataframe called df 







#d1=pd.read_csv("Dataset/Training/Features_Variant_1.csv")
#d1.columns=col_list
#d2=pd.read_csv("Dataset/Training/Features_Variant_2.csv")
#d2.columns=col_list
#d3=pd.read_csv("Dataset/Training/Features_Variant_3.csv")
#d3.columns=col_list
#d4=pd.read_csv("Dataset/Training/Features_Variant_4.csv")
#d4.columns=col_list
#d5=pd.read_csv("Dataset/Training/Features_Variant_5.csv")
#d5.columns=col_list
#
#
## In[91]:
#
#
#frames_main = [d1 , d2 , d3 , d4 , d5]
#df = pd.concat(frames_main)
#print (df.shape)
#df.head()

################################################################################################################################################################
# In[92]:

df.columns=col_list
df_train,df_test = train_test_split(df,train_size=0.7,random_state=42)
column_list1=['CC2','Base_time','Post_share_count','c3','c8','c18','CC1','CC4','Post_length_char_count','CC5']
x_train=df_train[column_list1]
y_train=df_train['Target_variable']
scaler.fit(x_train)
x_train_sc=scaler.transform(x_train)
x_test=df_test[column_list1]
y_test=df_test['Target_variable']
scaler.fit(x_test)
x_test_sc=scaler.transform(x_test)
print(x_train.shape,',',x_test.shape,',',y_train.shape,',',y_test.shape)


# In[93]:


df_summ=pd.DataFrame(columns=['Models           ','Dataset','R-sq','RMSE','MAE'])


# Random Forest Model

# In[97]:


rf=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=12, n_jobs=1,
           oob_score=False, random_state=42, verbose=0, warm_start=False)
rf.fit(x_train_sc, y_train)
filename = 'Models/RF_model.sav'
pickle.dump(rf, open(filename, 'wb'))


# Random Forest on training Dataset

# In[98]:


y_train_pred=rf.predict(x_train_sc)
r2=r2_score(y_train,y_train_pred)
mae=mean_absolute_error(y_train,y_train_pred)
rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
mod='Random Forest'
dataset='Training'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# Random Forest on testing dataset

# In[99]:


y_test_pred=rf.predict(x_test_sc)
r2=r2_score(y_test,y_test_pred)
mae=mean_absolute_error(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
mod='Random Forest'
dataset='Testing'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# Neural Networks

# In[100]:


mlp = MLPRegressor(activation='relu', alpha=1e-06, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(15, 15, 15), learning_rate='constant',
       learning_rate_init=0.001, max_iter=50, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
mlp.fit(x_train_sc,y_train)
filename = 'Models/NN_model.sav'
pickle.dump(mlp, open(filename, 'wb'))


# Neural Networks on training dataset

# In[101]:


y_train_pred=mlp.predict(x_train_sc)
r2=r2_score(y_train,y_train_pred)
mae=mean_absolute_error(y_train,y_train_pred)
rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
mod='Neural Network'
dataset='Training'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# Neural Networks on testing dataset

# In[102]:


y_test_pred=mlp.predict(x_test_sc)
r2=r2_score(y_test,y_test_pred)
mae=mean_absolute_error(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
mod='Neural Network'
dataset='Testing'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# KNN regressor

# In[103]:


knn = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=1, n_neighbors=2, p=2,
          weights='uniform')
knn.fit(x_train_sc,y_train)
filename = 'Models/KNN_model.sav'
pickle.dump(knn, open(filename, 'wb'))


# KNN on training Dataset

# In[104]:


y_train_pred=knn.predict(x_train_sc)
r2=r2_score(y_train,y_train_pred)
mae=mean_absolute_error(y_train,y_train_pred)
rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
mod='KNN'
dataset='Training'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# KNN on testing dataset

# In[105]:


y_test_pred=knn.predict(x_test_sc)
r2=r2_score(y_test,y_test_pred)
mae=mean_absolute_error(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
mod='KNN'
dataset='Testing'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# Linear Regression Model

# In[94]:


lm=linear_model.LinearRegression()
lm.fit(x_train_sc,y_train)
filename = 'Models/Linear_model.sav'
pickle.dump(lm, open(filename, 'wb'))


# Linear Regression on training dataset

# In[95]:


y_train_pred=lm.predict(x_train_sc)
r2=r2_score(y_train,y_train_pred)
mae=mean_absolute_error(y_train,y_train_pred)
rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
mod='Linear Regression'
dataset='Training'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# Linear Regression on Testing dataset

# In[96]:


y_test_pred=lm.predict(x_test_sc)
r2=r2_score(y_test,y_test_pred)
mae=mean_absolute_error(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
mod='Linear Regression'
dataset='Testing'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# SVR

# In[ ]:


svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svr.fit(x_train_sc,y_train)
filename = 'Models/SVR_model.sav'
pickle.dump(svr, open(filename, 'wb'))


# SVR on training dataset

# In[ ]:


y_train_pred=svr.predict(x_train_sc)
r2=r2_score(y_train,y_train_pred)
mae=mean_absolute_error(y_train,y_train_pred)
rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
mod='SVR'
dataset='Training'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# SVR on testing dataset

# In[ ]:


y_test_pred=svr.predict(x_test_sc)
r2=r2_score(y_test,y_test_pred)
mae=mean_absolute_error(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
mod='SVR'
dataset='Testing'
print(mod,' on ',dataset,' dataset ',' : ')
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))
data={'Models':mod,'Dataset':dataset,'R-sq':r2,'RMSE':rmse,'MAE':mae}
df_summ=df_summ.append(data,ignore_index=True)


# In[ ]:


print(df_summ)


# In[ ]:


df_summ.to_csv('Models/Summarry.csv',sep=',',index=False)

##########################################################################################
# zip the folder Models and upload it to s3




###########################################################################################
