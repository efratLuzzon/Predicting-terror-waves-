import pymysql
from sqlalchemy import create_engine

class DBHelper:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "root"
        self.db = "terror_db"
        self.con_string = 'mysql+mysqldb://' + self.user + ':' + self.password + '@' + self.host + ':3306/' \
                  + self.db + '?charset=utf8mb4'
        self.engine = create_engine(self.con_string)

    def __connect__(self):
        self.con = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()

    def __disconnect__(self):
        self.con.close()

    def fetch(self, query):
        self.__connect__()
        self.cur.execute(query)
        result = self.cur.fetchall()
        self.__disconnect__()
        return result

    def execute(self, query):
        self.__connect__()
        self.cur.execute(query)
        self.__disconnect__()

    def load_df(self,df, df_name):
        df.to_sql(name=df_name, con=self.engine, if_exists='replace', index=False, chunksize=1000,
                           method='multi')