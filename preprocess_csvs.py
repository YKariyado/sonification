import csv
import numpy as np
import pandas as pd
import matplotlib

files = []
data_csv = np.array([])
date = []

filepath = 'csv/'
names = ['data_temp.csv', 'data_dwpt.csv',
         'data_rh.csv', 'data_wind.csv', 'data_vp.csv']

# append file names.
for i in range(len(names)):
    files.append(names[i])


# # fill NaN-value with Mean
# for i in range(len(files)):
#     with open(filepath+files[i], encoding='utf8', newline='') as f:
#         csvreader = csv.reader(f)
#         each_row = []
#         for row in csvreader:
#             if len(row[1]) == 0:
#                 mean = (float(each_row[len(each_row)-1]) + float(each_row[len(each_row)-2]))/2.0
#                 each_row.append(mean)
#             else:
#                 each_row.append(float(row[1]))
            
#         each_row = np.array(each_row)

#         if i>0:
#             data_csv = np.vstack((data_csv, each_row))
#         else:
#             data_csv = np.append(data_csv, each_row)
#     print(data_csv.shape)


# fill NaN-value with using pandas interporation
col_list = ['temperature', 'dewpoint-temperature', 'humidity', 'wind velocity', 'vapor pressure']

for i in range(len(files)):
    with open(filepath+files[i], encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        each_row = []
        for row in csvreader:
            if len(row[1]) == 0:
                each_row.append(row[1])
            else:
                each_row.append(float(row[1]))

        each_row = np.array(each_row)

        if i>0:
            data_csv = np.vstack((data_csv, each_row))
        else:
            data_csv = np.append(data_csv, each_row)
    print(data_csv.shape)

data_csv = data_csv.T

# make data_csv as data frame
df1 = pd.DataFrame(data=data_csv, columns=col_list)
df1.replace("", np.nan, inplace=True)
df2 = df1.astype(float)

# print(df2.isnull().sum())
print(df2.info)
print(df2.describe())

with open('csv/data_temp.csv', encoding='utf8', newline='') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        date.append(row[0])

df2.insert(0, "date", date, True)

# # print something
# print('-----Data Frame-----')
# print(df2)
# print('-----Data Type-----')
# print(type(df2))
# print()

# interpolation
df3 = df2.interpolate()
print(df3.describe())
# save to csv file
df3.to_csv('csv/meteorological.csv')
print('saved!')



# # write
# with open('csv/data.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(data_csv.T)

# np.save(filepath+'aizuwakamatsu_weather', data_csv.T)

# print('saved!')



# # # check
# # loaded_csv = np.load('csv/aizuwakamatsu_weather.npy')

# # for row in range(5):
# #     print(loaded_csv[row])


# # # check 2
# df_check = pd.read_csv('csv/meteorological.csv')
# numpy_list = df_check.to_numpy()

# print(numpy_list)
# print(numpy_list.shape)
