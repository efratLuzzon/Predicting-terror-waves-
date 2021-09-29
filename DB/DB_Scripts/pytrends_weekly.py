from pytrends.request import TrendReq
import pandas as pd

country = 'Israel'  # can be changed

# Execute the TrendReq method by passing the host language (hl) and timezone (tz) parameters
pytrend = TrendReq(hl='ar', tz=120)  # can be changed

# Build list of keywords
colnames = ["keywords"]
df = pd.read_csv("indicative_keywords.csv", names=colnames)
# df = pd.read_csv("non_indicative_keywords.csv", names=colnames)
df2 = df["keywords"].values.tolist()
df2.remove("Keywords")

for year in range(2004, 2022):
    dataset = []
    for x in range(0, len(df2)):
        keyword = [df2[x]]
        if year == 2004:
            time_frame = str(year) + '-01-01 ' + str(year + 1) + '-01-06'
        elif year == 2021:
            time_frame = str(year - 1) + '-12-26 ' + str(year) + '-04-08'
        else:
            time_frame = str(year - 1) + '-12-26 ' + str(year + 1) + '-01-06'
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
    file_name = 'indicative_weekly_' + country + '_' + str(year) + '.csv'
    # file_name = 'non_indicative_weekly_' + country + '_' + str(year) + '.csv'
    result.to_csv(file_name, encoding='utf-8-sig')
