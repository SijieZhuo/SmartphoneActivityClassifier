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
import tsfresh.feature_selection
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import Main


class FeatureExtractionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        combine_btn = tk.Button(self, text="combine", command=lambda: combine_data())
        combine_btn.pack()

        analyse_btn = tk.Button(self, text="Analyse", command=lambda: combined2())
        analyse_btn.pack()

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"))
        back_btn.pack()

        selection_btn = tk.Button(self, text="selection", command=lambda: feature_selection())
        selection_btn.pack()

        test_btn = tk.Button(self, text="test", command=lambda: test_selected())
        test_btn.pack()


def combine_data():
    filename_list = []
    # get file name
    for file in os.listdir("Data/DataForAnalysation"):
        if file.endswith(".csv"):
            path = os.getcwd() + "\Data\DataForAnalysation\\" + file
            print(path)
            filename_list.append(path)

    result_data = None
    result_target = None
    index = 0

    for filepath in filename_list:
        file = pd.read_csv(filepath)

        # file["id"] = count
        # count = count + 1

        activity = get_target((file))[1]
        print(activity)
        sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))

        for window in sectioned_data:
            indexColumn = np.array([[index]] * len(window))
            window1 = np.append(indexColumn, window, axis=1)

            with open("Data/TrainingSet/datax.csv", 'a', newline='') as writeDataFile:
                writer = csv.writer(writeDataFile)
                writer.writerows(window1)
            writeDataFile.close()

            with open("Data/TrainingSet/targetx.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([[index, activity]])
            writeTargetFile.close()

            index = index + 1


def combined2():
    df = pd.read_csv("Data/TrainingSet/dataz.csv")
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]

    df2 = df.drop(columns=["activity"])
    df2 = df2.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    # print(df2)

    # result = tsfresh.feature_selection.relevance.calculate_relevance_table(df2, df["activity"])

    result = extract_features(df2, column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=ComprehensiveFCParameters(),
                              n_jobs=8, show_warnings=False, profile=False)

    with open("Data/TrainingSet/datay.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows([result.columns])
        writer.writerows(result.values)
    writeTargetFile.close()
    print("done")


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
        print(activity)
        sectioned_data = sliding_window(file, Main.window_size, int(Main.window_size / 2))

        for window in sectioned_data:
            # window = window[:, 1:-2]
            data = feature_extraction2(window)
            data = np.insert(data.values, 0, count)

            with open("Data/TrainingSet/data.csv", 'a', newline='') as writeDataFile:
                writer = csv.writer(writeDataFile)
                writer.writerows([data])
            writeDataFile.close()

            with open("Data/TrainingSet/target.csv", 'a', newline='') as writeTargetFile:
                writer = csv.writer(writeTargetFile)
                writer.writerows([[count, activity]])
            writeTargetFile.close()

        print("write")


def feature_selection():
    data = pd.read_csv("Data/TrainingSet/datay.csv")
    target = pd.read_csv("Data/TrainingSet/targetz.csv")
    # target.columns = ['index', 'target']
    print(target['target'])
    relevance_table = calculate_relevance_table(data, target['target'], fdr_level=0.0001)
    relevant_features = relevance_table[relevance_table.relevant].feature
    print(relevant_features)

    with open("Data/TrainingSet/features.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows([relevant_features])
    writeTargetFile.close()
    return


def test_selected():
    featureCSV = pd.read_csv("Data/TrainingSet/features.csv")

    print(len(featureCSV.columns))

    features = tsfresh.feature_extraction.settings.from_columns(featureCSV)
    print(features)
    print(type(features))
    print(features.keys())

    df = pd.read_csv("Data/TrainingSet/dataz.csv")
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

    with open("Data/TrainingSet/final.csv", 'w', newline='') as writeTargetFile:
        writer = csv.writer(writeTargetFile)
        writer.writerows(result.values)
    writeTargetFile.close()


def feature_extraction4(data):

    featureCSV = pd.read_csv("Data/TrainingSet/features.csv")
    features = tsfresh.feature_extraction.settings.from_columns(featureCSV)

    #df = pd.DataFrame(data)
    indexColumn = np.array([[0]] * len(data))
    df = np.append(indexColumn, data, axis=1)
    df = pd.DataFrame(df)
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"]
    print(df)
    df = df.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    result = extract_features(df[['id','time', 'accX']], column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=features.get('accX'),
                              n_jobs=8, show_warnings=False, profile=False)
    for datatype in ("accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ"):
        r = extract_features(df[['id','time', datatype]], column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=features.get(datatype),
                              n_jobs=8, show_warnings=False, profile=False)

        result = pd.merge(result,r,left_index=True,right_index=True)
    print(result)
    return result



def feature_extraction3(data):
    df = pd.DataFrame(data)
    df.columns = ["time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity", "id"]
    df = df[["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]]

    featureCSV = pd.read_csv("Data/TrainingSet/features.csv")
    features = tsfresh.feature_extraction.settings.from_columns(featureCSV)

    df2 = df.drop(columns=["activity"])
    df2 = df2.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    # print(df2)

    # result = tsfresh.feature_selection.relevance.calculate_relevance_table(df2, df["activity"])

    result = extract_features(df2, column_id='id', column_sort='time',
                              default_fc_parameters=features
                              )

    if not os.path.exists("Data/TrainingSet/data.csv"):
        with open("Data/TrainingSet/data.csv", 'w', newline='') as writeTargetFile:
            writer = csv.writer(writeTargetFile)
            writer.writerows([result])
        writeTargetFile.close()
    if not os.path.exists("Data/TrainingSet/target.csv"):
        with open("Data/TrainingSet/target.csv", 'w', newline='') as writeTargetFile:
            writer = csv.writer(writeTargetFile)
            writer.writerows([['index', 'target']])
        writeTargetFile.close()

    print(result)
    return result


def feature_extraction2(data):
    df = pd.DataFrame(data)
    df.columns = ["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]
    # df = df[["id", "time", "accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]]

    df2 = df.drop(columns=["activity"])
    df2 = df2.astype(
        {"accX": np.float32, "accY": np.float32, "accZ": np.float32, "rotX": np.float32, "rotY": np.float32,
         "rotZ": np.float32, "graX": np.float32, "graY": np.float32, "graZ": np.float32})

    # print(df2)

    # result = tsfresh.feature_selection.relevance.calculate_relevance_table(df2, df["activity"])

    result = extract_features(df2, column_id='id', column_sort='time', impute_function=impute,
                              default_fc_parameters=ComprehensiveFCParameters(),
                              n_jobs=8, show_warnings=False, profile=False)

    if not os.path.exists("Data/TrainingSet/datay.csv"):
        with open("Data/TrainingSet/data.csv", 'w', newline='') as writeTargetFile:
            writer = csv.writer(writeTargetFile)

            writer.writerows(result)
        writeTargetFile.close()
    '''
    if not os.path.exists("Data/TrainingSet/targety.csv"):
        with open("Data/TrainingSet/target.csv", 'w', newline='') as writeTargetFile:
            writer = csv.writer(writeTargetFile)
            writer.writerows([['index', 'target']])
        writeTargetFile.close()
    '''

    print(result)
    return result


def feature_extraction1(data):
    column_mean = pd.DataFrame(data).mean(axis=0)
    # column_sd = pd.DataFrame(data).std(axis=0)
    # column_varience = pd.DataFrame(data).var(axis=0)
    # column_min = pd.DataFrame(data).min(axis=0)
    # column_max = pd.DataFrame(data).max(axis=0)
    # column_mean_absolute_deviation = pd.DataFrame(data).mad(axis=0)
    # column_iqr = iqr(data, axis=0)
    column_ara = average_resultant_acceleration(data)
    # column_skewness = skew(data, axis=0)
    column_skewness = pd.DataFrame(data).skew(axis=0)
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
    return inputdata[inputdata.columns[10]]
