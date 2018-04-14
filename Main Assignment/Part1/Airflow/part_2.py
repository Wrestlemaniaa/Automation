from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Following are defaults which can be overridden later on
default_args = {
    'owner': 'jerin.rajan',
    'depends_on_past': False,
    'start_date': datetime(2018, 4, 10),
    'email': ['jerinrajan23@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('ADS2', default_args=default_args,schedule_interval='*/150 0 * * *')

# t1, t2, t3 and t4 are examples of tasks created using operators

t2 = BashOperator(
    task_id='part1',
    bash_command='python /home/ec2-user/airflow/dags/part_1.py',
    dag=dag)

t3 = BashOperator(
    task_id='part2',
    bash_command='python /home/ec2-user/airflow/dags/part_2.py',
    dag=dag)

t4 = BashOperator(
    task_id='part3',
    bash_command='python /home/ec2-user/airflow/dags/part_3.py',
    dag=dag)


t1 = BashOperator(
    task_id='Connected',
    bash_command='echo "ADS Assignment 3"',
    dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
t4.set_upstream(t3)

