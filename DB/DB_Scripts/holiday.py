import holidays
import pandas as pd
import datetime
import numpy as np
import sys


def calc_easter(year):
    "Returns Easter as a date object."
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1
    return datetime.date(year, month, day)


if __name__ == "__main__":
    years_list = range(1970, 2020)
    if sys.argv[1] == "jew":
        holidays_list = ["Passover", "Memorial Day", "Independence Day", "Lag B'Omer", "Shavuot", "Rosh Hashanah", "Yom Kippur",
                         "Sukkot", "Hanukkah", "Purim"]
        holiday_dates = holidays.Israel(years=years_list).items()
        file_name = "jewish_holidays.csv"

    if sys.argv[1] == "muslim":
        holidays_list = ["Eid al-Fitr", "Arafat Day", "Islamic New Year", "Sacrifice Feast",
                         "Prophet Muhammad's Birthday"]
        holidays_list_columns = ["Eid al-Fitr", "Arafat Day", "Islamic New Year", "Sacrifice Feast",
                         "Prophet Muhammad's Birthday", "Ramadan"]
        holiday_dates_egypt = holidays.Egypt(years=years_list).items()
        holiday_dates_Turkey = holidays.Turkey(years=years_list).items()
        # Merge the two dictitmes
        holiday_dates = holiday_dates_egypt | holiday_dates_Turkey
        #holiday_dates = holiday_dates_Turkey
        file_name = "muslim_holidays.csv"


    if sys.argv[1] == "chris":
        years_list = range(1970, 2020)
        holidays_list = ["New Year's Day", "Christmas Day"]
        holiday_dates = holidays.EuropeanCentralBank(years=years_list).items()
        file_name = "chris_holidays.csv"
        # holidays_list_columns is used for creating dataframe
        holidays_list_columns = ["New Year's Day", "Christmas Day", "Easter"]

    # Prepare dates column
    start_date = datetime.datetime(year=1970, month=1, day=1)
    end_date = datetime.datetime(year=2019, month=12, day=31)
    dates_list = pd.date_range(start_date, end_date)

    # Convert dates to string type
    str_dates_list = []
    for date in dates_list:
        date = str(date).split()[0]
        str_dates_list.append(date)

    # Create dataframe
    df = pd.DataFrame(np.zeros((len(str_dates_list), len(holidays_list_columns)), int), str_dates_list, holidays_list_columns)
    df.head()

    # Update dataframe
    for date, holiday in holiday_dates:
        holiday_name = [item for item in holidays_list if item in holiday]
        if holiday_name != []:
            df.at[str(date), holiday_name] = 1

    # changes for chris holidays
    if sys.argv[1] == "chris":
        easter_dates = []
        for year in years_list:
            easter_dates.append(calc_easter(year))

        # Update dataframe with easter dates
        for date in easter_dates:
                df.at[str(date), "Easter"] = 1

    # changes for islam holidays
    if sys.argv[1] == "muslim":
        ramadan_dates_list = []
        ramadan_df = pd.read_excel("updated_ramadan dates.xlsx")
        for index, row in ramadan_df.iterrows():
            day, month, year = row['first day of ramadan'].split('.')
            start_date = datetime.date(int(year), int(month), int(day))
            day, month, year = row['ramadan end date'].split('.')
            end_date = datetime.date(int(year), int(month), int(day))
            delta = end_date - start_date  # as timedelta

            for i in range(delta.days + 1):
                day = start_date + datetime.timedelta(days=i)
                ramadan_dates_list.append(day)

        # Update dataframe
        for date in ramadan_dates_list:
            df.at[str(date), "Ramadan"] = 1

    #df.to_excel(file_name)
    df.to_csv(file_name)
    #"""
