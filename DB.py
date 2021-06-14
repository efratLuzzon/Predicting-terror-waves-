import pymysql

class DBHelper:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "root"
        self.db = "terror_db"

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