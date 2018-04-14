import pandas as pd
import numpy as np
import pickle
import urllib.request
import sklearn
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import shutil
import logging 
import boto.s3
from boto.s3.key import Key
def aws_connect(ak,sak):
    if not ak or not sak:
        logging.warning('Access Key and Secret Access Key not provided!!')
        print('Access Key and Secret Access Key not provided!!')
        exit()
    
    AWS_ACCESS_KEY_ID = ak
    AWS_SECRET_ACCESS_KEY = sak
    
#    try:
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)
    print(AWS_ACCESS_KEY_ID,',',AWS_SECRET_ACCESS_KEY)  
    print("Connected to S3")
    print(conn)
    bucket1 = conn.get_bucket('adspickle1')
    srcFileName = 'Models'
    k = Key(bucket1,srcFileName)
    #Get the contents of the key into a file 
    k.get_contents_to_filename('Models.zip')
#
#    except:
#        logging.info("Amazon keys are invalid!!")
#        print("Amazon keys are invalid!!")
#        exit()

def credentials():
    credentials = {}
    with open('Usernames.txt', 'r') as f:
    	for line in f:
      	  user, pwd, url = line.strip().split(';')
      	  lst=[pwd,url]
      	  credentials[user] = lst
    	return credentials

def read_file(path):
    df=pd.read_csv(path)
    return df

def header_col():
    col_list=['Page_popularity','Page_visited_no_of_times','Page_talking_about','Page_category','c1','c2','c3','c4','c5','c6','c7','c8',
         'c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','CC1','CC2','CC3','CC4','CC5',
         'Base_time','Post_length_char_count','Post_share_count','Post_promoted','Time_target','Sunday_post','Monday_post',
         'Tuesday_post','Wednesday_post','Thrusday_post','Friday_post','Saturday_post','Sunday_base','Monday_base','Tuesday_base','Wednesday_base',
         'Thrusday_base','Friday_base','Saturday_base']
    return col_list

def col_selected():
    column_list1=['CC2','Base_time','Post_share_count','c3','c8','c18','CC1','CC4','Post_length_char_count','CC5']
    return column_list1
    
def zip_file(a):
    shutil.make_archive(a,'zip',a)

def unzip_file(a,b):
    shutil.unpack_archive(a,extract_dir=b)

def del_directory(a):
    shutil.rmtree(a)

def form_output(dff):
    lm=0
    rf=0
    mlp=0
    knn=0
    svr=0
    y=[]
    x=pd.DataFrame(data=[dff],columns=col_selected())
    scaler = StandardScaler()
    scaler.fit(x)
    x_test_sc=scaler.transform(x)
    print(x.shape)
    filename = 'Models/Models/Linear_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    lm=np.round(mod.predict(x_test_sc),0)
    filename = 'Models/Models/RF_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    rf=np.round(mod.predict(x_test_sc),0)
    filename = 'Models/Models/NN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    mlp=np.round(mod.predict(x_test_sc),0)
    filename = 'Models/Models/KNN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    knn=np.round(mod.predict(x_test_sc),0)
    filename = 'Models/Models/SVR_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    svr=np.round(mod.predict(x_test_sc),0)
    y=[rf,mlp,knn,lm,svr] 
    df=pd.DataFrame(data=[y],columns=['Random Forest','Neural Networks','KNN   ','Linear reg','SVR   '])
    return x,df

def model_run(df):
    #df.columns=pd.DataFrame(data=dff,columns=header_col())
    df.columns=header_col()
    print (df.shape)
    scaler = StandardScaler()
    x=df[col_selected()]
    print(x.head())
    scaler.fit(x)
    x_test_sc=scaler.transform(x)
    print(x.shape)
    d1=df.copy()
    filename = 'Models/Models/Linear_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d1['Target_variable']=mod.predict(x_test_sc)
    d1['Target_variable']=d1['Target_variable'].round(decimals=0)
    d1.to_csv('Output/Linear.csv',sep=',',index=False)
    d2=df.copy()
    filename = 'Models/Models/RF_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d2['Target_variable']=mod.predict(x_test_sc)
    d2['Target_variable']=d2['Target_variable'].round(decimals=0)
    d2.to_csv('Output/RF.csv',sep=',',index=False)
    d3=df.copy()
    filename = 'Models/Models/NN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d3['Target_variable']=mod.predict(x_test_sc)
    d3['Target_variable']=d3['Target_variable'].round(decimals=0)
    d3.to_csv('Output/NN.csv',sep=',',index=False)
    d4=df.copy()
    filename = 'Models/Models/KNN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d4['Target_variable']=mod.predict(x_test_sc)
    d4['Target_variable']=d4['Target_variable'].round(decimals=0)
    d4.to_csv('Output/KNN.csv',sep=',',index=False)
    d5=df.copy()
    filename = 'Models/Models/SVR_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d5['Target_variable']=mod.predict(x_test_sc)
    d5['Target_variable']=d5['Target_variable'].round(decimals=0)
    d5.to_csv('Output/SVR.csv',sep=',',index=False)
    d6=pd.read_csv("Models/Models/Summarry.csv")
    #del_directory('Models')
    zip_file('Output')
    return d2,d6

def rest_run(df):
    #df.columns=pd.DataFrame(data=dff,columns=header_col())
    #df.columns=header_col()
    print (df.shape)
    scaler = StandardScaler()
    x=df[col_selected()]
    print(x.head())
    scaler.fit(x)
    x_test_sc=scaler.transform(x)
    print(x.shape)
    d1=df.copy()
    filename = 'Models/Models/Linear_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d1['Target_variable']=mod.predict(x_test_sc)
    d1['Target_variable']=d1['Target_variable'].round(decimals=0)
    d1.to_csv('Output/Linear.csv',sep=',',index=False)
    d2=df.copy()
    filename = 'Models/Models/RF_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d2['Target_variable']=mod.predict(x_test_sc)
    d2['Target_variable']=d2['Target_variable'].round(decimals=0)
    d2.to_csv('Output/RF.csv',sep=',',index=False)
    d3=df.copy()
    filename = 'Models/Models/NN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d3['Target_variable']=mod.predict(x_test_sc)
    d3['Target_variable']=d3['Target_variable'].round(decimals=0)
    d3.to_csv('Output/NN.csv',sep=',',index=False)
    d4=df.copy()
    filename = 'Models/Models/KNN_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d4['Target_variable']=mod.predict(x_test_sc)
    d4['Target_variable']=d4['Target_variable'].round(decimals=0)
    d4.to_csv('Output/KNN.csv',sep=',',index=False)
    d5=df.copy()
    filename = 'Models/Models/SVR_model.sav'
    mod = pickle.load(open(filename, 'rb'))
    d5['Target_variable']=mod.predict(x_test_sc)
    d5['Target_variable']=d5['Target_variable'].round(decimals=0)
    d5.to_csv('Output/SVR.csv',sep=',',index=False)
    d6=pd.read_csv("Models/Models/Summarry.csv")
    zip_file('Output')
    return d2,d6
