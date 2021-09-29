from pytrends.request import TrendReq
import pandas as pd

country = 'Israel'  # can be changed

# Execute the TrendReq method by passing the host language (hl) and timezone (tz) parameters
pytrend = TrendReq(hl='ar', tz=120)  # can be changed

# Build list of keywords
colnames = ["keywords"]
file = pd.read_csv("indicative_keywords.csv", names=colnames)
# file = pd.read_csv("non_indicative_keywords.csv", names=colnames)
keywords = file["keywords"].values.tolist()
keywords.remove("Keywords")

for year in range(2004, 2022):
    for half in range(1, 3):
        dataset = []
        for x in range(0, len(keywords)):
            keyword = [keywords[x]]
            if half == 1:
                time_frame = str(year) + '-01-01 ' + str(year) + '-06-30'
            else:
                time_frame = str(year) + '-07-01 ' + str(year) + '-12-31'
            # Build the payload
            pytrend.build_payload(kw_list=keyword, cat=0, timeframe=time_frame, geo='IL')  # can be changed
                                                                                           # 'IL' - Israel
                                                                                           # 'PS' - Palestine
            # Store 'interest over time' information in 'data'
            data = pytrend.interest_over_time()
            if not data.empty:
                data = data.drop(labels=['isPartial'], axis='columns')
                dataset.append(data)

        result = pd.concat(dataset, axis=1)
        file_name = 'indicative_daily_' + country + '_' + str(year) + '_' + str(half) + '.csv'
        # file_name = 'non_indicative_daily_' + country + '_' + str(year) + '_' + str(half) + '.csv'
        result.to_csv(file_name, encoding='utf-8-sig')
        print(file_name)
        if year == 2021:
            break
