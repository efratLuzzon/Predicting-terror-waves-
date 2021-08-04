from datetime import timedelta
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def update(loss_df, start_wave, end_wave):
    delta = timedelta(days=1)
    while start_wave <= end_wave:
        loss_df.at[start_wave, "class"] = 1
        start_wave += delta
    return loss_df

def classify_data(loss_df):
    loss_df["class"] = 0
    days = []

    in_wave = False
    start_counr_end = False
    counter = 0
    start_wave = None
    end_wave = None
    for day in loss_df.iterrows():
        loss = day[1][0]
        timestamp = day[1][1]
        if (loss > 0.1):
            # start wave if didnt start
            if not in_wave:
                in_wave = True
                start_wave = timestamp

            else:
                # stop counting 30 days of decrease if in wave
                if start_counr_end:
                    counter = 0
                    start_counr_end = False
                    end_wave = None
        else:
            if in_wave:
                # start count 30 days if didnt start
                if not start_counr_end:
                    end_wave = timestamp
                    start_counr_end = True
                counter += 1
                # if 30 days of decrease passed end wave
                if counter >= 30:
                    if start_wave < end_wave - timedelta(days=21):
                        loss_df = update(loss_df, start_wave, end_wave)
                        print(str(start_wave.date()) + " - " + str(end_wave.date()))
                        days.append(start_wave.date())
                        days.append(end_wave.date())
                    in_wave = False
                    end_wave = None
                    counter = 0
                    start_counr_end = False
    days_df = pd.DataFrame(days, columns=["result"])
    days_df = pd.to_datetime(days_df["result"])
    return days_df


if __name__ == "__main__":
    loss_df = pd.read_csv("df.csv")
    loss_df.index = loss_df["timestamp"]
    loss_df["timestamp"] = pd.to_datetime(loss_df["timestamp"])

    classify_data = classify_data(loss_df)


    plt.figure(figsize=(20, 10))
    sns.set_style("darkgrid")
    ax = sns.distplot(loss_df["loss"], bins=100, label="Frequency")
    ax.set_title("Frequency Distribution | Kernel Density Estimation")
    ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
    plt.axvline(1.80, color="k", linestyle="--")
    plt.legend()

    plt.figure(figsize=(20, 10))
    ax = sns.lineplot(x="timestamp", y="loss", data=loss_df, color='g', label="Anomaly Score")
    ax.set_title("Anomaly Confidence Score vs Timestamp")
    ax.set(ylabel="Anomaly Confidence Score", xlabel="Timestamp")

    for i in range(0, len(classify_data), 2):
        c = 'b'
        if i % 4 == 0:
            c = 'r'
        plt.axvline(classify_data[i], color=c)
        plt.axvline(classify_data[i + 1], color=c)

    plt.legend()
    plt.show()