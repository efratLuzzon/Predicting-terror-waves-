from zope.interface import Interface


class DBI(Interface):
    def __connect__():
        """connect to DB"""

    def __disconnect__():
        """disconnect from DB"""

    def fetch(query, params):
        """fetch a query"""

    def load_df(df, df_name):
        """load dataframe to DB"""
