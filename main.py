import pandas as pd
import numpy as np
from functools import reduce

''' Tele CF '''
video_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\FINAL_RESULTS\Mohammad_MasterSheet_FrailtyIndex_V1.csv"
sensor_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\FINAL_RESULTS\Mohammad_MasterSheet_FrailtyIndex_sensors.csv"

''' Icampers '''
# video_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\ICAMP_Members_Results\FINAL_Mohammad_FI_MasterSheet\Mohammad_MasterSheet_FrailtyIndex.csv"
# sensor_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\ICAMP_Members_Results\FINAL_Mohammad_FI_MasterSheet\Mohammad_MasterSheet_FrailtyIndex_sensors_V1.csv"

''' Phase 1 '''
# video_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase I\Results\Mohammad_Results\Mohammad_Final_FI\Mohammad_MasterSheet_FrailtyIndex.csv"
# sensor_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase I\Results\Mohammad_Results\Mohammad_Final_FI\Mohammad_MasterSheet_FrailtyIndex_sensors_V1.csv"

''' Former youngs '''
# video_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\former_young_data\analysis\Mohammad_FI_MasterSheet\Mohammad_MasterSheet_FrailtyIndex.csv"
# sensor_file_location = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\former_young_data\analysis\Mohammad_FI_MasterSheet\Mohammad_MasterSheet_FrailtyIndex_sensors_V1.csv"

video_data = pd.read_csv(video_file_location, error_bad_lines=False)
sensor_data = pd.read_csv(sensor_file_location, error_bad_lines=False)
# print(video_data)
# print(sensor_data)
df = pd.merge(video_data, sensor_data, on="Subject_IDs")
print(df)
# print(list(df.columns))
# print(len(list(df.columns)))
all_columns = list(df.columns)
print(all_columns)
# print(all_columns[1:90])
# print(all_columns[90:])
video_cols = all_columns[1:90]
sensor_cols = all_columns[90:]


def dis_calc(x1, x2):
    if x1 < x2:
        a = np.linspace(x1, x2, 100, endpoint=True)
        x1_new = a[0]
        x2_new = a[40]
        return x1_new, x2_new
    else:
        a = np.linspace(x2, x1, 100, endpoint=True)
        x1_new = a[99]
        x2_new = a[60]
        return x1_new, x2_new


def applier(f1, f2, df):
    new_df = pd.DataFrame()
    # print('*' * 100)
    new_df[f'{f1}_'] = df[[f1, f2]].apply(lambda x: dis_calc(x[f1], x[f2]), axis=1)
    # print(new_df[f'{f1}_'])
    new_df[[f'{f1}_', f'{f2}_']] = pd.DataFrame(new_df[f'{f1}_'].tolist(), index=df.index)
    return new_df


# print(dis_calc(65050, 40000))
video_cols
sensor_cols
print(video_cols)
print(sensor_cols)
df_list = []
for i in range(len(video_cols)):
    # print('*' * 100)
    df_n = applier(video_cols[i], sensor_cols[i], df)
    df_list.append(df_n)
    # print(df_n)

# reduce(lambda x, y: pd.merge(x, y), df_list)
# np.concatenate(df.col2.values)
a = df_list[0] + df_list[1]
# print(a)
# print(pd.concat(df_list, axis=1))
final_df = pd.concat(df_list, axis=1)
print(final_df)
# print(df['Subject_IDs'])
final_df = pd.concat([df['Subject_IDs'], final_df], axis=1)
final_df = final_df.set_index('Subject_IDs')
# final_df.to_csv(r'Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\FINAL_RESULTS\updated_video_sensor.csv')
# final_df.to_csv(r'Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\ICAMP_Members_Results\FINAL_Mohammad_FI_MasterSheet\iCAMP_Members_Feb.csv')
# final_df.to_csv(r'Z:\Projects BCM\H-43917 Image based Frailty\Phase I\Results\Mohammad_Results\Mohammad_Final_FI\Phase1_Mar.csv')
# final_df.to_csv(r'Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\former_young_data\analysis\Mohammad_FI_MasterSheet\former_young_Feb.csv')

print(final_df)
# print()
print(list(final_df.loc[['TEL007']].to_numpy()[0]))
print(final_df.loc[['TEL007']])
final_df.loc[['TEL007']].to_csv(r'Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\FINAL_RESULTS\TEL007.csv')
