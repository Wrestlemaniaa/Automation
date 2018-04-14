
# coding: utf-8

# In[5]:


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


# In[2]:


col_list=['Page_popularity','Page_visited_no_of_times','Page_talking_about','Page_category','c1','c2','c3','c4','c5','c6','c7','c8',
         'c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','CC1','CC2','CC3','CC4','CC5',
         'Base_time','Post_length_char_count','Post_share_count','Post_promoted','Time_target','Sunday_post','Monday_post',
         'Tuesday_post','Wednesday_post','Thrusday_post','Friday_post','Saturday_post','Sunday_base','Monday_base','Tuesday_base','Wednesday_base',
         'Thrusday_base','Friday_base','Saturday_base','Target_variable']
d1=pd.read_csv("Dataset/Training/Features_Variant_1.csv")
d1.columns=col_list
d2=pd.read_csv("Dataset/Training/Features_Variant_2.csv")
d2.columns=col_list
d3=pd.read_csv("Dataset/Training/Features_Variant_3.csv")
d3.columns=col_list
d4=pd.read_csv("Dataset/Training/Features_Variant_4.csv")
d4.columns=col_list
d5=pd.read_csv("Dataset/Training/Features_Variant_5.csv")
d5.columns=col_list


# In[3]:


frames_main = [d1 , d2 , d3 , d4 , d5]
df = pd.concat(frames_main)
print (df.shape)
df.head()


# In[4]:


df_train,df_test = train_test_split(df,train_size=0.7,random_state=42)
column_list1=['CC2','Base_time','Post_share_count','c3','c8','c18','CC1','CC4','Post_length_char_count','CC5']
x_train=df_train[column_list1]
y_train=df_train['Target_variable']
scaler.fit(x_train)
x_train_sc=scaler.transform(x_train)
x_test=df_test[column_list1]
#x_test=x_test.iloc[:100,:]
y_test=df_test['Target_variable']
#y_test=y_test.iloc[:100]
scaler.fit(x_test)
x_test_sc=scaler.transform(x_test)
print(x_train.shape,',',x_test.shape,',',y_train.shape,',',y_test.shape)


# In[6]:


filename = 'SVR_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[7]:
print("Traning")
y_train_pred=loaded_model.predict(x_train_sc)
print("R2   :",r2_score(y_train,y_train_pred))
print("MAE  :",mean_absolute_error(y_train,y_train_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_train,y_train_pred)))


print("Testing")
y_test_pred=loaded_model.predict(x_test_sc)
print("R2   :",r2_score(y_test,y_test_pred))
print("MAE  :",mean_absolute_error(y_test,y_test_pred))
print("RMSE :",np.sqrt(mean_squared_error(y_test,y_test_pred)))

