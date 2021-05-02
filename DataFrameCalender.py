import calendar

import pandas as pd


class DataFrameCalender:
    @staticmethod
    def create_empty_dates(start_day_in_week, start_year, end_year):
        dates = []
        c = calendar.Calendar(start_day_in_week)
        for j in range(start_year, end_year + 1):
            for i in range(1, 13):
                iter_days = c.itermonthdates(j, i)
                for day in iter_days:
                    if day.month is not i:
                        continue
                    dates.append(day)
        dates = pd.DataFrame(dates, columns=['date'])
        return dates

    @staticmethod
    def set_date_time_index(data, column_name, column_to_convert):
        data[column_name] = pd.to_datetime(column_to_convert)
        data.index = data[column_name]
        return data
