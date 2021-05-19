import holidays
import pandas as pd
import datetime
import numpy as np
import sys

if __name__ == "__main__":
    years_list = range(1970, 2021)
    if sys.argv[1] == "jew":
        holidays_list = ["Passover", "Memorial Day", "Independence Day", "Lag B'Omer", "Shavuot", "Rosh Hashanah", "Yom Kippur",
                         "Sukkot", "Hanukkah", "Purim"]
        holiday_dates = holidays.Israel(years=years_list).items()
        file_name = "jewish_holidays.xlsx"

    if sys.argv[1] == "muslim":
        holidays_list = ["Eid al-Fitr", "Arafat Day", "Islamic New Year",
                         "Ramadan Feast", "Sacrifice Feast", "Prophet Muhammad's Birthday", ]
        holiday_dates_egypt = holidays.Egypt(years=2020).items()
        holiday_dates_Turkey = holidays.Turkey(years=2020).items()
        print(holiday_dates_egypt)
        print(holiday_dates_Turkey)
        # Merge the two dictitmes
        holiday_dates = holiday_dates_egypt | holiday_dates_Turkey
        holiday_dates = holiday_dates_Turkey
        file_name = "muslim_holidays.xlsx"

    """
    if sys.argv[1] == "chris":
        holidays_list = holidays.Norway(years=2018).values()
        holiday_dates_Italy = holidays.Norway(years=years_list).items()
        holiday_dates = holiday_dates_Italy
        file_name = "chris_holidays.xlsx"
    """

    # Prepare dates column
    start_date = datetime.datetime(year=1970, month=1, day=1)
    end_date = datetime.datetime(year=2020, month=12, day=31)
    dates_list = pd.date_range(start_date, end_date)

    # Convert dates to string type
    str_dates_list = []
    for date in dates_list:
        date = str(date).split()[0]
        str_dates_list.append(date)

    # Create dataframe
    df = pd.DataFrame(np.zeros((len(str_dates_list), len(holidays_list)), int), str_dates_list, holidays_list)
    df.head()

    # Update dataframe
    for date, holiday in holiday_dates:
        holiday_name = [item for item in holidays_list if item in holiday]
        if holiday_name != []:
            df.at[str(date), holiday_name] = 1

    df.to_excel(file_name)
