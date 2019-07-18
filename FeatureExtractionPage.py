import csv
import os
import tkinter as tk
import pandas as pd
import numpy as np

import Main

class FeatureExtractionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        analyse_btn = tk.Button(self, text="Analyse", command=lambda: analyse_data())
        analyse_btn.pack()


def analyse_data():
    filename_list = []


    for file in os.listdir("Data/DataForAnalysation"):
        if file.endswith(".csv"):
            path = os.getcwd() + "\Data\DataForAnalysation\\" + file
            print(path)
            filename_list.append(path)

    result_data = None
    result_target = None

    for filepath in filename_list:
        file = pd.read_csv(filepath)

        activity = get_target((file))[1]
        output_data = []
        sectioned_data = sliding_window(pd.DataFrame(get_data(file)), Main.window_size, int(Main.window_size / 2))
        for window in sectioned_data:
            data = feature_extraction(window)
            output_data.append(data)
            #print(data)
            #print("")

        length = len(output_data)
        output_target = [[activity]] * length

        if result_data is None:
            result_data = output_data
        else:
            result_data = np.concatenate((result_data, output_data))
        if result_target is None:
            result_target = output_target
        else:
            result_target = np.concatenate((result_target, output_target))

    with open("Data/TrainingSet/data.csv", 'w', newline='') as writeDataFile:
        writer = csv.writer(writeDataFile)
        writer.writerows(result_data)
        print("write")
    writeDataFile.close()

    with open("Data/TrainingSet/target.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows(result_target)
        print("write")
    writeTargetFile.close()


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


def feature_extraction(data):

    column_mean = data.mean(axis=0)
    column_sd = data.std(axis=0)
    column_varience = data.var(axis=0)
    column_mean_absolute_deviation = pd.DataFrame(data).mad(axis=0)
    column_ara = average_resultant_acceleration(data)

    features = np.concatenate((column_mean,column_sd, column_varience, column_mean_absolute_deviation, column_ara))
    return features


def average_resultant_acceleration(data):
    acc_sum = 0.0
    gyro_sum = 0.0
    magnet_sum = 0.0
    for column in data:
        acc = np.math.sqrt(column[0] ** 2 + column[1] ** 2 + column[2] ** 2)
        gyro = np.math.sqrt(column[3] ** 2 + column[4] ** 2 + column[5] ** 2)
        magnet = np.math.sqrt(column[6] ** 2 + column[7] ** 2 + column[8] ** 2)

        acc_sum = acc_sum + acc
        gyro_sum = gyro_sum + gyro
        magnet_sum = magnet_sum + magnet

    average_acc = acc_sum / len(data)
    average_gyro = gyro_sum / len(data)
    average_magnet = magnet_sum / len(data)

    return [average_acc, average_gyro, average_magnet]
