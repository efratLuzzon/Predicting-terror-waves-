import calendar
import glob
import pandas as pd
from DataFrameCalender import DataFrameCalender
from Definition import GTD
from DB_Scripts.TerrorData import TerrorAttackData


def write_default_values_google_trends(indicative_words, prefix_column, path_write_file):
    dates_list = DataFrameCalender.create_empty_dates(calendar.SUNDAY, 1970, 2003)
    # write default value
    dict_columns_name = dict()
    for word in indicative_words:
        dict_columns_name[prefix_column + word] = [-1 for i in range(len(dates_list))]

    words_df = pd.DataFrame(dict_columns_name)
    words_df = pd.concat([dates_list, words_df], axis=1)
    words_df = DataFrameCalender.set_date_time_index(words_df, 'date', words_df['date'])
    words_df.to_csv(path_write_file, index=False, mode='a', header=True, encoding='utf-8')


def write_unified_google_trends_by_country(indicative_words, prefix_name, path_file, path_write_data):
    # load real values
    files = glob.glob(path_file + "\\*.csv")
    for file in files:
        file_data = pd.read_csv(file)

        data = pd.DataFrame({'date': file_data['date']})
        data = DataFrameCalender.set_date_time_index(data, 'date', file_data["date"])
        # get and sorted columns names
        columns_names = file_data._series.keys()
        file_data = DataFrameCalender.set_date_time_index(file_data, 'date', file_data['date'])
        columns_names = list(columns_names)[1:]  # pop 'date' column
        columns_names = sorted(columns_names)

        # concat all columns by order - if the word doesn't exist in the original file
        # put 0 to all columns field.
        i = 0
        for word in indicative_words:
            # the word exist
            if i < len(columns_names) and word == columns_names[i]:
                data = pd.concat([data, file_data[columns_names[i]]], axis=1)
                # rename the column with prefix "israel or falastin"
                data.rename(columns={word: prefix_name + word}, inplace=True)
                i += 1
            else:
                d = pd.DataFrame(0, index=file_data['date'], columns=[prefix_name + word])
                data = pd.concat([data, d], axis=1)
        # append order data to file
        data.to_csv(path_write_data, index=False, mode='a', header=False)


def write_unified_google_trends():
    # load keywords
    indicative_words = pd.read_csv("../load_data\\Google Trends - Daily Data\\indicative_keywords.csv")
    indicative_words = sorted(indicative_words['Keywords'])
    prefix_column = ['Israel - ', 'Palestine - ']
    path_to_write = ['google_trends_israel.csv', 'google_trends_palestine.csv']
    path_read_files = ['../load_data\\Google Trends - Daily Data\\Israel\\indicative',
                       '../load_data\\Google Trends - Daily Data\\Palestine\\indicative']
    for i in range(2):
        write_default_values_google_trends(indicative_words, prefix_column[i], path_to_write[i])
        write_unified_google_trends_by_country(indicative_words=indicative_words, prefix_name=prefix_column[i],
                                               path_file=path_read_files[i], path_write_data=path_to_write[i])


def insert_data_files_to_matrix_data_model(data_model):
    files = glob.glob("load_data/*.csv")
    for file in files:
        pd_data = pd.read_csv(file)
        data_model = concat_table_to_matrix_model(pd_data, data_model)
    return data_model


def concat_table_to_matrix_model(table, matrix):
    num_days_in_matrix = len(matrix)
    columns_name = table.columns
    for c in columns_name:
        assert len(table[c]) == num_days_in_matrix
        if c == 'date' or c == 'dates':
            table[c] = pd.to_datetime(table[c])
            table.index = table[c]
        else:
            matrix = pd.concat([matrix, table[c]], axis=1)
    return matrix


if __name__ == "__main__":
    #write_unified_google_trends()
    terror_data = TerrorAttackData("../gtd1970-2019_clean.csv", 1970, 2019, GTD.SELECTED_COUNTRY, info=True)
    data_model = terror_data.load_data()
    data_model = insert_data_files_to_matrix_data_model(data_model)
    data_model.to_csv("matrix.csv")
