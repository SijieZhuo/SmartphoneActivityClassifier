import tkinter as tk
from tkinter import filedialog
from sklearn.neural_network import MLPClassifier
import pandas as pd
import Main
import FeatureExtractionPage
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns


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

        classify_btn = tk.Button(self, text="Classify", command=lambda: classify_btn_hit(self))
        classify_btn.pack()

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"))
        back_btn.pack()

        test_btn = tk.Button(self, text="test", command=lambda: self.check_data())
        test_btn.pack()

    def check_data(self):
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())

        corr = data_file.corr()
        #corr = corr.values
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
                        classify_data.append(float(phone_data[i]))
                        # phone_data[i].astype(float32)

                # float_data = [float(i) for i in phone_data]
                separated = [classify_data[x:x + 9] for x in range(0, len(classify_data), 9)]
                print(separated)
                for row in separated:
                    self.data_window.append(row)

                if len(self.data_window) == Main.window_size:
                    self.current_processed_data.data = FeatureExtractionPage.feature_extraction1(
                        np.array(self.data_window))

                    del self.data_window[:(int(Main.window_size / 2))]

            self.classify_window(self.current_processed_data.data)

    def classify_window(self, data):
        if self.is_classifying is True:
            print([data])
            print([[data]])

            prediction = self.clf.predict(data)
            print(prediction)


def browse_btn_hit(folder_path):
    # Allow user to select a directory and store it in global var
    # called folder_path
    filename = filedialog.askopenfilename()
    folder_path.set(filename)
    print(filename)


def classify_btn_hit(page):
    data_file = pd.read_csv(page.data_file_path.get())
    target_file = pd.read_csv(page.target_file_path.get())
    page.is_classifying = True
    page.clf.fit(data_file.values, target_file.values.ravel())




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
