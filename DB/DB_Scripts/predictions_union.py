import os
import glob
import pandas as pd
os.chdir("predictions")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
predictions_df = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
predictions_df.to_csv("predictions.csv", index=False, encoding='utf-8-sig')