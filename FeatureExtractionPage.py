import csv
import os
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsfresh
import tsfresh.feature_selection
from scipy.stats import kurtosis, iqr
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from scipy.signal import find_peaks

import Main


class FeatureExtractionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # flow of ts feature extraction is : combine_data()  ->  feature_extracton_ts()  -> feature_selection()
        #  -> generate_final_features_ts()

        # button for time frequency domain feature extraction
        tf_btn = tk.Button(self, text="Time/Frequency extraction", command=lambda: time_frequency_feature_extraction(), width=25)
        tf_btn.grid(row=1, column=1, pady=10)

        # selection_btn = tk.Button(self, text="ts selection", command=lambda: feature_selection())
        # selection_btn.pack()

        # button for tsfresh feature extraction
        ts_btn = tk.Button(self, text="generate ts features", command=lambda: ts_feature_extraction(), width=25)
        ts_btn.grid(row=2, column=1, pady=10)

        test_btn = tk.Button(self, text="test features", command=lambda: test_feature(), width=25)
        test_btn.grid(row=3, column=1, pady=10)

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"), width=25)
        back_btn.grid(row=4, column=1, pady=10)
        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_rowconfigure((0, 5), weight=1)


# extract data from each of the csv files, and combine them into one big csv file
def combine_data():
    filename_list = []
    # get file name
    for file in os.listdir("Data/DataForAnalysation"):
        if file.endswith(".csv"):
            path = os.getcwd() + "\Data\DataForAnalysation\\" + file
            print(path)
            filename_list.append(path)

    index = 0

    with open("Data/TrainingSet/data_all.csv", 'w', newline='') as writeDataFile:
        writer = csv.writer(writeDataFile)
        writer.writerows(
            [['id', 'time', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ', 'activity']])
    writeDataFile.close()

    with open("Data/TrainingSet/target_all.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows([['index', 'activity']])
    writeTargetFile.close()

    for filepath in filename_list:
        file = pd.read_csv(filepath)

        activity = get_target((file))[1]
        print(activity)
        sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))

        for window in sectioned_data:
            indexColumn = np.array([[index]] * len(window))
            window1 = np.append(indexColumn, window, axis=1)

            with open("Data/TrainingSet/data_all.csv", 'a', newline='') as writeDataFile:
                writer = csv.writer(writeDataFile)
                writer.writerows(window1)
            writeDataFile.close()

            with open("Data/TrainingSet/target_all.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([[index, activity]])
            writeTargetFile.close()

            index = index + 1


# first step of extract ts features
# reading the big csv file been generated form combine_data(), and extract all the possible features in tsfresh library
def feature_extraction_ts():
    df = pd.read_csv("Data/TrainingSet/data_all.csv")
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]

    df2 = df.drop(columns=["activity"])
    df2 = df2.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    result = extract_features(df2, column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=ComprehensiveFCParameters(),
                              n_jobs=8, show_warnings=False, profile=False)

    with open("Data/TrainingSet/data_ts.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows([result.columns])
        writer.writerows(result.values)
    writeTargetFile.close()
    print("done")


# analysing the extracted features, and generate a list of features that are relevant to the data
def feature_selection():
    data = pd.read_csv("Data/TrainingSet/data_ts.csv")
    target = pd.read_csv("Data/TrainingSet/target_all.csv")
    target.columns = ['index', 'target']
    print(target['target'])
    relevance_table = calculate_relevance_table(data, target['target'], fdr_level=0.0001)
    relevant_features = relevance_table[relevance_table.relevant].feature
    print(relevant_features)

    with open("Data/TrainingSet/features.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows([relevant_features])
    writeTargetFile.close()
    return


# generate the final sets of features that would be used for the ML training
def generate_final_features_ts():
    featureCSV = pd.read_csv("Data/TrainingSet/features.csv")

    features = tsfresh.feature_extraction.settings.from_columns(featureCSV)

    df = pd.read_csv("Data/TrainingSet/data_all.csv")
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]

    result = extract_features(df[['id', 'time', 'accX']], column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=features.get('accX'),
                              n_jobs=8, show_warnings=False, profile=False)
    for datatype in ("accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"):
        r = extract_features(df[['id', 'time', datatype]], column_id='id', column_sort='time', impute_function=impute,
                             default_fc_parameters=features.get(datatype),
                             n_jobs=8, show_warnings=False, profile=False)

        result = pd.merge(result, r, left_index=True, right_index=True)
    print(result)

    with open("Data/TrainingSet/data_ts_final.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows(result.values)
    writeTargetFile.close()

    df = pd.read_csv("Data/TrainingSet/target_all.csv", skiprows=0)
    with open("Data/TrainingSet/target_ts_final.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows(df.values)
    writeTargetFile.close()


# method used for realtime feature extraction
# input is a window of raw data, and the output is the feature generated
def feature_extraction_ts_realtime(data):
    featureCSV = pd.read_csv("Data/TrainingSet/features.csv")
    features = tsfresh.feature_extraction.settings.from_columns(featureCSV)

    # df = pd.DataFrame(data)
    indexColumn = np.array([[0]] * len(data))
    df = np.append(indexColumn, data, axis=1)
    df = pd.DataFrame(df)
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"]
    df = df.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    result = extract_features(df[['id', 'time', 'accX']], column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=features.get('accX'),
                              n_jobs=8, show_warnings=False, profile=False)
    for datatype in ("accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"):
        r = extract_features(df[['id', 'time', datatype]], column_id='id', column_sort='time', impute_function=impute,
                             default_fc_parameters=features.get(datatype),
                             n_jobs=8, show_warnings=False, profile=False)

        result = pd.merge(result, r, left_index=True, right_index=True)
    return result


def ts_feature_extraction():
    combine_data()
    feature_extraction_ts()
    feature_selection()
    generate_final_features_ts()


# calculate the features in the selected folder and save the extracted file in TrainingSet
def time_frequency_feature_extraction():
    filename_list = []
    # get file name
    for file in os.listdir("Data/DataForAnalysation"):
        if file.endswith(".csv"):
            path = os.getcwd() + "\Data\DataForAnalysation\\" + file
            filename_list.append(path)

    for filepath in filename_list:
        print(filepath)
        file = pd.read_csv(filepath)
        file.columns = ["time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]

        activity = get_target(file)[1]
        del file['time']
        del file['activity']
        file = file.astype(
            {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
             "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

        sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))

        for window in sectioned_data:
            window = pd.DataFrame(window)
            window.columns = ["accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"]
            window = window.astype(
                {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
                 "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})
            result = feature_extraction1(window)

            with open("Data/TrainingSet/data_tf.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([result])
            writeTargetFile.close()

            with open("Data/TrainingSet/target_tf.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([[activity]])
            writeTargetFile.close()


# manually calculate time and frequency domain features
def feature_extraction1(data):
    data = pd.DataFrame(data)

    column_mean = data.mean(axis=0)
    column_mean = column_mean[:6]  # mean for megnetometer data was high corrolated

    column_sd = pd.DataFrame(data).std(axis=0)

    column_varience = data.var(axis=0)

    column_min = data.min(axis=0)
    column_min = column_min[:3]  # min for gyro and megnetometer data was high corrolated

    column_max = data.max(axis=0)
    column_max = column_max[:3]  # max for megnetometer data was high corrolated

    # column_mean_absolute_deviation = data.mad(axis=0)   #high coor

    # column_iqr = iqr(data, axis=0)                     # high coor

    column_ara = average_resultant_acceleration(data)  # 3 columns
    column_ara = column_ara[:1]  # ara  for gyro and megnetometer data was high corrolated

    column_skewness = data.skew(axis=0)

    column_kurtosis = kurtosis(data, axis=0)

    column_sma = sma(data)  # Signal magnitude area, 3 columns
    column_sma = column_sma[:1]  # sma  for gyro and megnetometer data was high corrolated

    column_energy = energy(data)  # high coor
    column_energy = column_energy[:3]  # energy for gyro and megnetometer data was high corrolated

    column_zrc = zero_crossing_rate(data)  # currently reduce accuracy

    column_no_peaks = no_peaks(data)

    features = np.concatenate(
        (column_mean, column_sd, column_varience, column_min, column_max,
         column_ara, column_skewness, column_kurtosis, column_sma, column_energy, column_zrc, column_no_peaks))

    return features


#  ======================= features =============================

def average_resultant_acceleration(data):
    data = data.values
    acc_sum = 0.0
    gyro_sum = 0.0
    magnet_sum = 0.0

    for row in data:
        acc = np.math.sqrt(row[0] ** 2 + row[1] ** 2 + row[2] ** 2)
        gyro = np.math.sqrt(row[3] ** 2 + row[4] ** 2 + row[5] ** 2)
        magnet = np.math.sqrt(row[6] ** 2 + row[7] ** 2 + row[8] ** 2)

        acc_sum = acc_sum + acc
        gyro_sum = gyro_sum + gyro
        magnet_sum = magnet_sum + magnet

    average_acc = acc_sum / len(data)
    average_gyro = gyro_sum / len(data)
    average_magnet = magnet_sum / len(data)

    return [average_acc, average_gyro, average_magnet]


def energy(data):
    return (data.values ** 2).sum(axis=0)


def sma(data):  # signal magnitude area
    data = data.values
    acc_sum = 0.0
    gyro_sum = 0.0
    magnet_sum = 0.0
    for row in data:
        acc = abs(row[0]) + abs(row[1]) + abs(row[2])
        gyro = abs(row[3]) + abs(row[4]) + abs(row[5])
        magnet = abs(row[6]) + abs(row[7]) + abs(row[8])

        acc_sum = acc_sum + acc
        gyro_sum = gyro_sum + gyro
        magnet_sum = magnet_sum + magnet

    average_acc = acc_sum / len(data)
    average_gyro = gyro_sum / len(data)
    average_magnet = magnet_sum / len(data)

    return [average_acc, average_gyro, average_magnet]


def zero_crossing_rate(data):
    zcr_list = []
    for col in data.values.T:
        col_zcr = ((col[:-1] * col[1:]) < 0).sum()
        zcr_list.append(col_zcr)

    return zcr_list


def no_peaks(data):
    list = []
    for col in data.values.T:
        peaks, _ = find_peaks(col)
        list.append(len(peaks))

    return list


#  ===================== end of features ==========================

def sliding_window(data, window_size, step_size):
    r = np.arange(len(data))
    s = r[::step_size]
    z = list(zip(s, s + window_size))
    g = lambda t: data.loc[t[0]:t[1]]
    result = map(g, z)
    result_set = []
    for item in result:
        result_set.append(item.values)
    while len(result_set[0]) != len(result_set[-1]):
        del result_set[-1]
    return result_set


def get_data(inputdata):
    return inputdata[inputdata.columns[0:9]]


def get_target(inputdata):
    return inputdata[inputdata.columns[-1]]


def test_feature():
    # file = pd.read_csv("Data/DataForAnalysation/2019_09_10_12_09_33_Walking_Watch.csv")
    #
    # file.columns = ["time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]
    #
    # del file['time']
    # del file['activity']
    # file = file.astype(
    #     {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
    #      "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})
    #
    # sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))
    # for window in sectioned_data:
    #     window = pd.DataFrame(window)
    #     output = iqr(window, axis=0)
    #
    #     print(output)

    file1 = pd.read_csv("Data/TrainingSet/data_ts.csv")
    file2 = pd.read_csv("Data/TrainingSet/data_ts_final.csv")

    print(len(file1.columns))
    print(len(file2.columns))
