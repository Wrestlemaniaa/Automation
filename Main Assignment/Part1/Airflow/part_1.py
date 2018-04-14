import pandas as pd

url1="https://s3-us-west-2.amazonaws.com/datadump101/Features_Variant_1.csv"
print("URL")
df =pd.read_csv(url1)
print("In")
df.to_csv('/home/ec2-user/airflow/dags/Features_Varient_1.csv',encoding='utf-8',index = False)
print("Out")
