import csv
import re


def is_suitable(sentence):
    weapdetails = str(sentence).lower()
    match_r = re.search(".*rocket*", weapdetails)
    match_m = re.search(".*mortar*", weapdetails)
    return not (match_r or match_m)


terror_attack_file = open("gtd1970-2019.csv", "r", encoding='ISO-8859-1')
x = open("gtd1970-2019_clean.csv", 'w', newline='', encoding='ISO-8859-1')
wr = csv.writer(x, quoting=csv.QUOTE_ALL)

terror_attack_reader = csv.reader(terror_attack_file)
next(terror_attack_reader)  # for header

for row in terror_attack_reader:
    if row[7] == '97':
        if not (row[35] == "Military") and not (row[84] == "Projectile (rockets, mortars, RPGs, etc.)") \
                and is_suitable(row[97]) and is_suitable(
            row[18]):  # weapdetails and summery not contain rocket and mortar
            wr.writerow(row)
    elif not (row[35] == "Military" or row[35] == "Police"):
        wr.writerow(row)
