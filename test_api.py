import pickle
import pandas as pd
import requests
import base64

df = pd.read_csv("confusion_matrix_2002.csv",index_col=False)
#print(df)
pickled = pickle.dumps(df)
pickled_b64 = base64.b64encode(pickled)

#r = requests.post('http://localhost:5000/ModelDateResult', data={'test_year': "2002"})
r = requests.post('http://localhost:5000/ModelDateResult', json={'test_year': 2002})
if r.ok:
    print("ok")
else:
    print(r.status_code)