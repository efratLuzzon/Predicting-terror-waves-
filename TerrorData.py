import csv
from datetime import datetime, timedelta
from datetime import date
import calendar
import numpy as np
import pandas as pd

from DataFrameCalender import DataFrameCalender
from Definition import GTD

'''
The script loads the data from GTD for all years for specific countries(by id)
Arguments - 
* start_year + end_year - the years to examine. 
* id_countries - list of the countries that need to load their data.
* start_day - start day of the calender, default Sunday.
'''


class TerrorAttackData:
    def __init__(self, path_data_file, start_year, end_year, id_countries, start_day=calendar.SUNDAY, info=False):
        self.__start_year = start_year
        self.__end_year = end_year
        self.__start_day_in_week = start_day
        self.__path_file = path_data_file
        self.__id_countries = id_countries
        self.__with_information = info

    '''
    create for each country list of dates, num attact per day - defualt zero
    and empty dict of information about the attack.
    '''

    @staticmethod
    def insert_bool_type_to_data_df(data, type_to_insert, index, initalize_list):
        if type_to_insert in data.columns:
            data.at[index, type_to_insert] = 1
        else:
            new_column = pd.DataFrame(initalize_list, columns=[type_to_insert], dtype='int')
            data[type_to_insert] = new_column.values
        return data

    @staticmethod
    def update_data_value_sum_type(data, index, type_to_update):
        if type_to_update is not "":
            if data.at[index, 'num_deaths'] == -1:
                data.at[index, 'num_deaths'] += (int(type_to_update) + 1)
            else:
                data.at[index, 'num_deaths'] += int(type_to_update)
        else:
            data.at[index, 'num_deaths'] = -1
        return data

    def load_data(self):
        dates = DataFrameCalender.create_empty_dates(start_day_in_week=self.__start_day_in_week,
                                                     start_year=self.__start_year,
                                                     end_year=self.__end_year)
        zero_list = [np.int64(0) for i in range(len(dates))]
        new_column = pd.DataFrame(zero_list, columns=['num_attack'], dtype='int')
        if self.__with_information:
            default_list = [-1 for i in range(len(dates))]
            new_column = pd.DataFrame({'num_attack': pd.Series(zero_list, dtype='int'),
                                       'succeed': pd.Series(zero_list, dtype='int'),
                                       'num_deaths': pd.Series(default_list, dtype='int'),
                                       'num_wounded': pd.Series(default_list, dtype='int')})
        data = pd.concat([dates, new_column], axis=1)
        data = DataFrameCalender.set_date_time_index(data, 'date', data['date'])
        terror_attack_file = open(self.__path_file, "r", encoding='ISO-8859-1')
        terror_attack_reader = csv.reader(terror_attack_file)
        for row in terror_attack_reader:
            if row[GTD.COUNTRY] == self.__id_countries:
                # Ignore unknown dates - day or month is '0'
                if int(row[GTD.MONTH]) is 0 or int(row[GTD.DAY]) is 0:
                    continue
                else:
                    current_date = date(int(row[GTD.YEAR]), int(row[GTD.MONTH]), int(row[GTD.DAY]))
                    current_date = pd.to_datetime(current_date)
                    data.at[current_date, 'num_attack'] += 1
                    if self.__with_information:
                        data.at[current_date, 'succeed'] |= np.int64(int(row[GTD.SUCCESS]))
                        data = self.update_data_value_sum_type(data, current_date, row[GTD.NUM_KILL])
                        data = self.update_data_value_sum_type(data, current_date, row[GTD.NUM_WOUND])
                        data = self.insert_bool_type_to_data_df(data, row[GTD.ATTACK_TYPE], current_date, zero_list)
                        data = self.insert_bool_type_to_data_df(data, row[GTD.WEAP_TYPE], current_date, zero_list)
        return data

    '''
    load data from GTD
    '''

    def num_days_from_start_date(self, d2):
        d1 = date(self.__start_year, 1, 1)
        return (d2 - d1).days

    '''
    
    
    def load_data_by_quarter(self, country_dict):
        terror_attack_file = open(self.__path_file, "r", encoding='ISO-8859-1')
        terror_attack_reader = csv.reader(terror_attack_file)
        for row in terror_attack_reader:
            if row[7] in self.__id_countries:
                # Ignore unknown dates - day or month is '0'
                if int(row[IDX_MONTH]) is 0 or int(row[IDX_DAY]) is 0:
                    #???
                    current_date = date(int(row[IDX_YEAR]), 1, 1)
                else:
                    quarter = get_quar(int(row[IDX_MONTH]))
                    current_date = date(int(row[IDX_YEAR]), int(quarter), int(1))
                num_months = self.num_quarter_from_start_date(current_date)
                if num_months >= 0:
                    id = row[IDX_ID_COUNTRY]
                    country_dict[id][num_months][1] += 1
                    if self.__with_information:
                        country_dict[id][num_months][2][row[0]] = row[27]
        return country_dict
    

    def create_empty_dates_month(self):
        dates = []
        for j in range(self.__start_year, self.__end_year + 1):
            for i in range(1, 13):
                c_date = date(int(j), int(i), int(1))
                if self.__with_information:
                    dates.append([c_date, 0, {}])
                else:
                    dates.append([c_date, 0])
        country_dict = self.create_dict_country(dates)
        return country_dict

    def create_empty_dates_quarter(self):
        dates = []
        for j in range(self.__start_year, self.__end_year + 1):
            for i in range(1, 5):
                c_date = date(int(j), int(i), int(1))
                if self.__with_information:
                    dates.append([c_date, 0, {}])
                else:
                    dates.append([c_date, 0])
        country_dict = self.create_dict_country(dates)
        return country_dict
    

    def create_empty_dates_week(self):
        dates = []
        dates.append([date(1969, 12, 29), 0])
        c = calendar.Calendar(self.__start_day_in_week)
        for j in range(self.__start_year, self.__end_year + 1):
            for i in range(1, 13):
                iter_days = c.itermonthdates(j, i)
                for day in iter_days:
                    week_day = day.weekday()
                    if day.month is not i:
                        continue
                    if self.__with_information:
                        dates.append([day, 0, {}])
                    # 0 is monday
                    elif week_day == 0:
                        dates.append([day, 0])
        country_dict = self.create_dict_country(dates)
        return country_dict

    def load_data_by_week(self, country_dict):
        terror_attack_file = open(self.__path_file, "r", encoding='ISO-8859-1')
        terror_attack_reader = csv.reader(terror_attack_file)
        for row in terror_attack_reader:
            if row[7] in self.__id_countries:
                # Ignore unknown dates - day or month is '0'
                if int(row[IDX_MONTH]) is 0 or int(row[IDX_DAY]) is 0:
                    # ???
                    current_date = date(int(row[IDX_YEAR]), 1, 1)
                else:
                    day_date = date(int(row[IDX_YEAR]), int(row[IDX_MONTH]), int(row[IDX_DAY]))
                    week_day = day_date.weekday()
                    week_date = day_date - timedelta(days=week_day)
                num_weeks = int(self.num_days_from_start_date(date(1969, 12, 29), week_date) / 7)
                if num_weeks >= 0:
                    id = row[IDX_ID_COUNTRY]
                    country_dict[id][num_weeks][1] += 1
                    if self.__with_information:
                        country_dict[id][num_weeks][2][row[0]] = row[27]
        return country_dict


    def load_data_by_month(self, country_dict):
        terror_attack_file = open(self.__path_file, "r", encoding='ISO-8859-1')
        terror_attack_reader = csv.reader(terror_attack_file)
        for row in terror_attack_reader:
            if row[7] in self.__id_countries:
                # Ignore unknown dates - day or month is '0'
                if int(row[IDX_MONTH]) is 0 or int(row[IDX_DAY]) is 0:
                    # ???
                    current_date = date(int(row[IDX_YEAR]), 1, 1)
                else:
                    current_date = date(int(row[IDX_YEAR]), int(row[IDX_MONTH]), int(1))
                num_months = self.num_months_from_start_date(current_date)
                if num_months >= 0:
                    id = row[IDX_ID_COUNTRY]
                    country_dict[id][num_months][1] += 1
                    if self.__with_information:
                        country_dict[id][num_months][2][row[0]] = row[27]
        return country_dict
    
    

    def num_days_from_start_date(self,d1, d2):
        return (d2 - d1).days

    def num_months_from_start_date(self, d2):
        d1 = date(self.__start_year, 1, 1)
        return d2.month - d1.month + 12 * (d2.year - d1.year)

    def num_quarter_from_start_date(self, d2):
        d1 = date(self.__start_year, 1, 1)
        return d2.month - d1.month + 4 * (d2.year - d1.year)
       
       
       def get_quar(month):
    if month in range(1, 4):
        return 1
    elif month in range(4, 7):
        return 2
    elif month in range(7, 10):
        return 3
    elif month in range(10, 13):
        return 4
    return 0

    '''
