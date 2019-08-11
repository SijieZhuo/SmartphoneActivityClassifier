import csv
import os
import tkinter as tk
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis, iqr
import _thread
from tsfresh import extract_features
import tsfresh
import Main


class FeatureExtractionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        analyse_btn = tk.Button(self, text="Analyse", command=lambda: analyse_data())
        analyse_btn.pack()

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"))
        back_btn.pack()


def analyse_data():
    filename_list = []
    # get file name
    for file in os.listdir("Data/DataForAnalysation"):
        if file.endswith(".csv"):
            path = os.getcwd() + "\Data\DataForAnalysation\\" + file
            print(path)
            filename_list.append(path)

    result_data = None
    result_target = None
    count = 0

    for filepath in filename_list:
        file = pd.read_csv(filepath)

        file["id"] = count
        count = count + 1

        activity = get_target((file))[1]
        sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))
        for window in sectioned_data:
            data = feature_extraction2(window)

            with open("Data/TrainingSet/data.csv", 'a', newline='') as writeDataFile:
                writer = csv.writer(writeDataFile)
                writer.writerows([data])
            writeDataFile.close()

            with open("Data/TrainingSet/target.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([[activity]])
            writeTargetFile.close()

        print("write")


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
    return inputdata[inputdata.columns[9]]


def feature_extraction2(data):
    df = pd.DataFrame(data)
    df.columns = ["time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity", "id"]
    df = df[["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]]

    df = df.drop(columns=["activity"])
    df = df.astype({"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
                    "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    result = extract_features(df, column_id='id', column_sort='time')
    print(result.values)
    with open("Data/TrainingSet/fe.csv", 'w', newline='') as writeDataFile:
        writer = csv.writer(writeDataFile)
        writer.writerows([result])
        writer.writerows(result.values)
    writeDataFile.close()
    print("done")
    return


def feature_extraction1(data):
    column_mean = pd.DataFrame(data).mean(axis=0)
    # column_sd = pd.DataFrame(data).std(axis=0)
    # column_varience = pd.DataFrame(data).var(axis=0)
    # column_min = pd.DataFrame(data).min(axis=0)
    # column_max = pd.DataFrame(data).max(axis=0)
    # column_mean_absolute_deviation = pd.DataFrame(data).mad(axis=0)
    # column_iqr = iqr(data, axis=0)
    column_ara = average_resultant_acceleration(data)
    column_skewness = skew(data, axis=0)
    column_kurtosis = kurtosis(data, axis=0)
    column_sma = sma(data)

    features = np.concatenate(
        (column_mean,
         column_ara, column_skewness, column_kurtosis, column_sma))
    return features


def average_resultant_acceleration(data):
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


def sma(data):
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
