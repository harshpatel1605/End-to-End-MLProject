import os
import sys
from src.mlproject.excption import CustomException
from src.mlproject.logger import logging
import pandas as pd
import pymysql

import pickle
import numpy as np

from dotenv import load_dotenv
load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')



def read_sql_data():
    logging.info("Reading MySQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )

        logging.info(f"Connection Established {mydb}")

        df = pd.read_sql_query('SELECT * FROM students',mydb)
        print(df.head())

        return df
        
    except Exception as e:
        raise CustomException(e,sys)
    
def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
