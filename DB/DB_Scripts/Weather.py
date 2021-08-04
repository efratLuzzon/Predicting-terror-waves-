import csv
import os
import datetime

AVERAGE = False
def convert_string_to_datetime(string_date):
    date_list = string_date.split("/")
    if len(date_list) == 1:
        date_list = string_date.split("-")
        if len(date_list) == 1:
            print("cann't convert to datetime " + string_date)
            exit(1)
    if int(date_list[0]) > 1000:  # year first
        date_day = datetime.datetime(int(date_list[0]), int(date_list[1]), int(date_list[2]))
    else:
        date_day = datetime.datetime(int(date_list[2]), int(date_list[1]), int(date_list[0]))
    return date_day


degrees = open("weather_years/degrees_1970_2019.csv", "r", encoding='ISO-8859-1')
reader_degrees = csv.reader(degrees)

perciption = open("weather_years/precipitation_1970_2019.csv", "r", encoding='ISO-8859-1')
reader_perciption = csv.reader(perciption)

weather_isreal = open("isreal_weather.csv", 'w', newline='', encoding='ISO-8859-1')
weather_isreal_writer = csv.writer(weather_isreal, quoting=csv.QUOTE_ALL)
weather_isreal_writer.writerow(["date", "avg_degree", "perciption"])

# for header
rain_list_day = next(reader_perciption)
next(reader_degrees)

rain_list_day = next(reader_perciption)
date_rain_day = convert_string_to_datetime(rain_list_day[0])


def get_degrees(max_degree, min_degree):
    aveg_degree = None
    if AVERAGE:
        if max_degree is not '-' and min_degree is not '-':
            aveg_degree = round((float(max_degree) + float(min_degree)) / 2, 2)
        elif max_degree is not '-':
            aveg_degree = round(float(max_degree), 2)
        elif min_degree is not '-':
            aveg_degree = round(float(min_degree), 2)
    elif max_degree is not '-':
        return round(float(max_degree), 2)
    return aveg_degree


for day in reader_degrees:
    date = convert_string_to_datetime(day[0])
    aveg = get_degrees(day[1], day[2])
    assert aveg is not None
    if date_rain_day is not None and date == date_rain_day:
        weather_isreal_writer.writerow([date, aveg, rain_list_day[1]])
        try:
            rain_list_day = next(reader_perciption)
            date_rain_day = convert_string_to_datetime(rain_list_day[0])
        except:
            date_rain_day = None
    else:
        weather_isreal_writer.writerow([date, aveg, 0])

degrees.close()
perciption.close()
weather_isreal.close()
