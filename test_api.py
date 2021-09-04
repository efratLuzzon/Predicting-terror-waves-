import pickle
import pandas as pd
import requests
import base64

df = pd.read_csv("wave_picture.csv",index_col=False, encoding = 'ISO-8859-8')
#print(df)
pickled = pickle.dumps(df)
pickled_b64 = base64.b64encode(pickled)
table_name = 'terror_wave_details'
#r = requests.get('http://localhost:5000/Features?year=2002')
#r = requests.post('http://localhost:5000/UploadFiles', data={'pickled_df': pickled_b64, 'table_name' : table_name})
#r = requests.post('http://localhost:5000/ModelDateResult', json={'pickled_df': pickled_b64, 'table_name' : table_name})
r = requests.post('http://localhost:5000/Anomalies', json={})
if r.ok:
    print("ok")
else:
    print(r.status_code)