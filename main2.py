import pandas as pd
import numpy as np
from functools import reduce

''' Tele CF '''
video_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\Frailty_index\results_all_TEL050_FM_BL_ST.xlsx"
sensor_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\Frailty_index\sensor_results_all_TEL050_FM_BL_ST.xlsx"


def dis_calc(x1, x2):
    # if x1 < x2:
    #     a = np.linspace(x1, x2, 100, endpoint=True)
    #     x1_new = a[5]
    #     x2_new = a[15]
    #     return x1_new, x2_new
    # else:
    #     a = np.linspace(x2, x1, 100, endpoint=True)
    #     x1_new = a[95]
    #     x2_new = a[85]
    #     return x1_new, x2_new
    return x1, x2

def applier(f1, f2, df):
    new_df = pd.DataFrame()
    print('*' * 100)
    # print(f1)
    # print(f2)
    # print(df[f1 + f2])
    new_df[f'{f1}_'] = df[f1 + f2].apply(lambda x: dis_calc(x[f1], x[f2]), axis=1)
    # print(new_df[f'{f1}_'])
    new_df[[f'{f1}_', f'{f2}_']] = pd.DataFrame(new_df[f'{f1}_'].tolist(), index=df.index)
    return new_df


video_data = pd.read_excel(video_file_location, sheet_name='Sheet1')
sensor_data = pd.read_excel(sensor_file_location, sheet_name='Sheet1')
# print(video_data)
# print(sensor_data)
df = pd.merge(video_data, sensor_data
              , on='Unnamed: 0'
              )
print(df)
# print(list(df.columns))
all_columns = list(df.columns)
video_cols = all_columns[1:102]
sensor_cols = all_columns[102:]
# print(video_cols)
# print(sensor_cols)

df_n = applier(video_cols, sensor_cols, df)
# print(list(df_n.columns))
print(df_n)
