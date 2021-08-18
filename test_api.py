import pickle
import pandas as pd
import requests
import base64

df = pd.read_csv("wave_picture.csv",index_col=False, encoding = 'ISO-8859-8')
#print(df)
pickled = pickle.dumps(df)
pickled_b64 = base64.b64encode(pickled)

r = requests.post('http://localhost:5000/Test', data={'pickled_df': pickled_b64})
#r = requests.post('http://localhost:5000/ModelDateResult', json={'test_year': 2002})
#r = requests.post('http://localhost:5000/Anomalies', json={})
if r.ok:
    print("ok")
else:
    print(r.status_code)