import sys
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np

if __name__ == "__main__":
    ramadan_dates_list = []
    ramadan_df = pd.read_excel("ramadan_dates.xlsx")
    for index, row in ramadan_df.iterrows():
        day, month, year = row['first day of ramadan'].split('.')
        start_date = date(int(year), int(month), int(day))
        day, month, year = row['ramadan end date'].split('.')
        end_date = date(int(year), int(month), int(day))
        delta = end_date - start_date  # as timedelta

        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            ramadan_dates_list.append(day)

    # Prepare dates column
    start_date = datetime(year=1970, month=1, day=1)
    end_date = datetime(year=2020, month=12, day=31)
    dates_list = pd.date_range(start_date, end_date)

    # Convert dates to string type
    str_dates_list = []
    for date in dates_list:
        date = str(date).split()[0]
        str_dates_list.append(date)

    # Create dataframe
    df = pd.DataFrame(np.zeros((len(str_dates_list), 1), int), str_dates_list, ["is_holiday"])
    df.head()

    # Update dataframe
    for date in ramadan_dates_list:
        df.at[str(date)] = 1

    df.to_excel("ramadan.xlsx")