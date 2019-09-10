import csv
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import FeatureExtractionPage
import Main


class ClassifyPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.data = controller.current_data
        self.data.bind_to(self.update_data)

        self.data_window = []
        self.current_processed_data = DataWindow()
        self.current_processed_data.data = [[]]
        self.current_processed_data.bind_to(self.classify_window)
        self.is_classifying = False

        self.method_text = tk.StringVar()
        self.method_text.set("Time/Frequency")

        self.data_file_path = tk.StringVar()
        data_label = tk.Label(self, textvariable=self.data_file_path)
        data_label.pack()
        data_path_btn = tk.Button(self, text="Browse data set", command=lambda: browse_btn_hit(self.data_file_path))
        data_path_btn.pack()

        self.target_file_path = tk.StringVar()
        target_label = tk.Label(self, textvariable=self.target_file_path)
        target_label.pack()
        target_path_btn = tk.Button(self, text="Browse target set",
                                    command=lambda: browse_btn_hit(self.target_file_path))
        target_path_btn.pack()

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.clf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=4, random_state=0)
        self.clf3 = neighbors.KNeighborsClassifier(15, weights='uniform')
        self.clf4 = svm.SVC(gamma='scale')

        self.method_btn = tk.Button(self, textvariable=self.method_text, command=lambda: self.select_classify_method())
        self.method_btn.pack()

        classify_btn = tk.Button(self, text="Classify", command=lambda: self.classify_btn_hit())
        classify_btn.pack()

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"))
        back_btn.pack()

        test_btn = tk.Button(self, text="test", command=lambda: self.check_data())
        test_btn.pack()

    def check_data(self):
        data_file = pd.read_csv(self.data_file_path.get())
        # target_file = pd.read_csv(self.target_file_path.get())

        corr = data_file.corr()
        # corr = corr.values
        print(corr)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(corr.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.columns)
        plt.show()

        corr = corr.values
        with open("Data/TrainingSet/corr.csv", 'w', newline='') as writeDataFile:
            writer = csv.writer(writeDataFile)
            writer.writerows(corr)
        writeDataFile.close()

    def update_data(self, data):

        if self.is_classifying is True:
            data2 = data.decode("utf-8")
            data3 = data2.replace(' ', '')
            data4 = data3.replace('[', '')
            phone_data = data4.replace(']', '').split(',')
            classify_data = []
            if len(phone_data) == 50:
                for i in range(0, len(phone_data)):
                    if i % 10 != 0:
                        phone_data[i] = float(phone_data[i])
                        # classify_data.append(float(phone_data[i]))
                        # phone_data[i].astype(float32)

                # float_data = [float(i) for i in phone_data]
                if self.method_text.get() == 'tsfresh':
                    print("tsfresh")
                    separated = [phone_data[x:x + 10] for x in range(0, len(phone_data), 10)]
                    for row in separated:
                        self.data_window.append(row)
                        if len(self.data_window) == Main.window_size:
                            self.current_processed_data.data = FeatureExtractionPage.feature_extraction_ts_realtime(
                                self.data_window)

                            del self.data_window[:(int(Main.window_size / 2))]

                            self.classify_window(self.current_processed_data.data)

                elif self.method_text.get() == 'Time/Frequency':
                    separated = [phone_data[x:x + 10] for x in range(0, len(phone_data), 10)]
                    for row in separated:
                        row = row[1:]
                        self.data_window.append(row)
                        if len(self.data_window) == Main.window_size:
                            self.current_processed_data.data = FeatureExtractionPage.feature_extraction1(
                                self.data_window)
                            del self.data_window[:(int(Main.window_size / 2))]
                            self.classify_window(self.current_processed_data.data)

    def classify_window(self, data):
        if self.is_classifying is True:
            if self.method_text.get() == 'tsfresh':
                data_to_predict = data.values
            elif self.method_text.get() == 'Time/Frequency':
                data_to_predict = [data]

            prediction = self.clf2.predict(data_to_predict)
            print(prediction)
            probability = self.clf2.predict_proba(data_to_predict)
            prediction_prob = {}
            for i in range(0, len(probability[0])):
                prediction_prob[self.clf2.classes_[i]] = probability[0][i]
            sorted_prob = sorted(prediction_prob.items(), key=lambda kv: kv[1], reverse=True)
            print(sorted_prob)

    def classify_btn_hit(self):
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())

        if self.method_text.get() == 'tsfresh':
            target_file.columns = ['index', 'target']

            self.clf.fit(data_file.values, target_file['target'])
            scoresm = cross_val_score(self.clf, data_file.values, target_file['target'], cv=5)
            print(scoresm.mean())

            self.clf2.fit(data_file.values, target_file['target'])
            scoresr = cross_val_score(self.clf2, data_file.values, target_file['target'], cv=5)
            print(scoresr.mean())

            self.clf3.fit(data_file.values, target_file['target'])
            scoresr = cross_val_score(self.clf3, data_file.values, target_file['target'], cv=5)
            print(scoresr.mean())

            self.clf4.fit(data_file.values, target_file['target'])
            scoresr = cross_val_score(self.clf4, data_file.values, target_file['target'], cv=5)
            print(scoresr.mean())

        elif self.method_text.get() == 'Time/Frequency':

            self.clf.fit(data_file.values, target_file)
            scoresm = cross_val_score(self.clf, data_file.values, target_file, cv=5)
            print(scoresm.mean())

            self.clf2.fit(data_file.values, target_file)
            scoresr = cross_val_score(self.clf2, data_file.values, target_file, cv=5)
            print(scoresr.mean())

            self.clf3.fit(data_file.values, target_file)
            scoresr = cross_val_score(self.clf3, data_file.values, target_file, cv=5)
            print(scoresr.mean())

            self.clf4.fit(data_file.values, target_file)
            scoresr = cross_val_score(self.clf4, data_file.values, target_file, cv=5)
            print(scoresr.mean())

        self.is_classifying = True

    def select_classify_method(self):
        if self.method_text.get() == "Time/Frequency":
            self.method_text.set("tsfresh")
        elif self.method_text.get() == "tsfresh":
            self.method_text.set("Time/Frequency")


def browse_btn_hit(folder_path):
    # Allow user to select a directory and store it in global var
    # called folder_path
    filename = filedialog.askopenfilename()
    folder_path.set(filename)
    print(filename)


class DataWindow(object):
    def __init__(self):
        self._data = []
        self._observers = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        for callback in self._observers:
            callback(self._data)

    def bind_to(self, callback):
        self._observers.append(callback)
