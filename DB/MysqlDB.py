import zope
from sqlalchemy import create_engine
import pymysql
from DB.DB_I import DBI


@zope.interface.implementer(DBI)
class MysqlDB():
    def __init__(self):
        self.__host = "localhost"
        self.__user = "root"
        self.__password = "root"
        self.__db = "terror_db"
        self.__con_string = 'mysql+mysqldb://' + self.__user + ':' + self.__password + '@' + self.__host + ':3306/' \
                            + self.__db + '?charset=utf8mb4'
        self.__engine = create_engine(self.__con_string)
        self.__con = None
        self.__cur = None

    def __connect__(self):
        self.__con = pymysql.connect(host=self.__host, user=self.__user, password=self.__password, db=self.__db,
                                     cursorclass=pymysql.cursors.
                                     DictCursor)
        self.__cur = self.__con.cursor()

    def __disconnect__(self):
        self.__con.close()

    def fetch(self, query, params=None):
        self.__connect__()
        self.__cur.execute(query, params)
        result = self.__cur.fetchall()
        self.__disconnect__()
        return result

    def load_df(self, df, df_name):
        df.to_sql(name=df_name, con=self.__engine, if_exists='replace', index=False, chunksize=1000,
                  method='multi')
