import csv
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
# from tsfresh.feature_selection.relevance import calculate_relevance_table
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

import statsmodels.api as sm

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

        self.combine_text = tk.StringVar()
        self.combine_text.set("Combine reading & scrolling")

        self.data_file_path = tk.StringVar()
        data_label = tk.Label(self, textvariable=self.data_file_path)
        data_label.grid(column=1, pady=3)
        data_path_btn = tk.Button(self, text="Browse data set", command=lambda: browse_btn_hit(self.data_file_path),
                                  width=25)
        data_path_btn.grid(column=1, pady=3)

        self.target_file_path = tk.StringVar()
        target_label = tk.Label(self, textvariable=self.target_file_path)
        target_label.grid(column=1, pady=3)
        target_path_btn = tk.Button(self, text="Browse target set",
                                    command=lambda: browse_btn_hit(self.target_file_path), width=25)
        target_path_btn.grid(column=1, pady=3)

        # ML models
        self.clf = MLPClassifier(solver='adam', activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,),
                                 random_state=0)
        self.clf2 = RandomForestClassifier(n_estimators=50, min_samples_split=2, criterion='entropy', random_state=0)
        self.clf3 = neighbors.KNeighborsClassifier(5)
        self.clf4 = svm.SVC(gamma='scale')
        self.clf5 = BaggingClassifier(neighbors.KNeighborsClassifier(5), max_samples=1.0, n_estimators=20,
                                      max_features=0.06)
        # current best, extremely randomized tree algorithm
        self.clf6 = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=4,
                                         criterion='gini', random_state=0)
        self.clf7 = AdaBoostClassifier(n_estimators=100)

        # selecting which method is used for classify data (time frequency / tsfresh)
        self.method_btn = tk.Button(self, textvariable=self.method_text, command=lambda: self.select_classify_method(),
                                    width=25)
        self.method_btn.grid(column=1, pady=3)

        # selecting whether combining reading and scrolling is applied or not
        self.combine_btn = tk.Button(self, textvariable=self.combine_text,
                                     command=lambda: self.select_label_combination(),
                                     width=25)
        self.combine_btn.grid(column=1, pady=3)

        # actual realtime classification
        classify_btn = tk.Button(self, text="Classify", command=lambda: self.classify_btn_hit(), width=25)
        classify_btn.grid(column=1, pady=3)

        # ploting the coorelation graph for the features
        coor_btn = tk.Button(self, text="Correlation plot", command=lambda: self.plot_coorelation(), width=25)
        coor_btn.grid(column=1, pady=3)

        # confusion matrix
        confusion_btn = tk.Button(self, text="Confusion matrix", command=lambda: self.confusion_matrix(), width=25)
        confusion_btn.grid(column=1, pady=3)

        # 5 fold cross validation for the classifiers
        validate_btn = tk.Button(self, text="Validate", command=lambda: self.validate_btn_hit(), width=25)
        validate_btn.grid(column=1, pady=3)

        # validate the model by randomising the target
        validate_random_btn = tk.Button(self, text="Validate by Random", command=lambda: self.validate_by_random(),
                                        width=25)
        validate_random_btn.grid(column=1, pady=3)

        # learning curve
        learning_curve_btn = tk.Button(self, text="Learning Curve", command=lambda: self.plot_learning_curve(),
                                        width=25)
        learning_curve_btn.grid(column=1, pady=3)

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"), width=25)
        back_btn.grid(column=1, pady=3)

        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_rowconfigure((0, 12), weight=1)

        # setup variables for the classify window
        self.window = tk.Toplevel(self)

        self.window_text1 = tk.StringVar()
        self.window_text2 = tk.StringVar()
        self.window_text3 = tk.StringVar()

        self.label1 = tk.Label(self.window, textvariable=self.window_text1)
        self.label2 = tk.Label(self.window, textvariable=self.window_text2)
        self.label3 = tk.Label(self.window, textvariable=self.window_text3)

        self.window_text1_accuracy = tk.StringVar()
        self.window_text2_accuracy = tk.StringVar()
        self.window_text3_accuracy = tk.StringVar()

        self.accuracy1 = tk.Label(self.window, textvariable=self.window_text1_accuracy)
        self.accuracy2 = tk.Label(self.window, textvariable=self.window_text2_accuracy)
        self.accuracy3 = tk.Label(self.window, textvariable=self.window_text3_accuracy)

        self.setup_window()

    def setup_window(self):
        """
        This function setup the configuration of the classification window
        """
        self.window.wm_title("Classification")
        self.window.geometry("600x400")
        self.window.withdraw()

        self.window_text1.set("prediction")
        self.window_text2.set("prediction")
        self.window_text3.set("prediction")

        self.label1.config(font=("Courier", 44))
        self.label2.config(font=("Courier", 33))
        self.label3.config(font=("Courier", 22))

        self.window_text1_accuracy.set("accuracy")
        self.window_text2_accuracy.set("accuracy")
        self.window_text3_accuracy.set("accuracy")

        self.accuracy1.config(font=("Courier", 20))
        self.accuracy2.config(font=("Courier", 20))
        self.accuracy3.config(font=("Courier", 20))

        self.label1.pack()
        self.accuracy1.pack()
        self.label2.pack()
        self.accuracy2.pack()
        self.label3.pack()
        self.accuracy3.pack()

    def update_window(self, prob):
        """
        this function updates the classification window in real time
        :param prob: the probability of each of the labels
        """
        firstPrediction = prob[0]
        secondPrediction = prob[1]
        thirdPrediction = prob[2]

        self.window_text1.set(firstPrediction[0].replace("_", " "))
        self.window_text2.set(secondPrediction[0].replace("_", " "))
        self.window_text3.set(thirdPrediction[0].replace("_", " "))

        self.window_text1_accuracy.set(firstPrediction[1])
        self.window_text2_accuracy.set(secondPrediction[1])
        self.window_text3_accuracy.set(thirdPrediction[1])

    def plot_coorelation(self):
        """
        This function generates
        """
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())
        data_file.astype(np.float32)
        corr = data_file.corr()
        target_file.columns = ['target']
        self.change_read_to_scroll(target_file)

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

    def update_data(self, data):
        """
        This function gather the real-time data from bluetooth and use them in feature extraction and classification
        :param data: raw data from bluetooth
        """
        if self.is_classifying is True:
            data2 = data.decode("utf-8")
            data3 = data2.replace(' ', '')
            data4 = data3.replace('[', '')
            phone_data = data4.replace(']', '').split(',')
            if len(phone_data) == 50:
                for i in range(0, len(phone_data)):
                    if i % 10 != 0:
                        phone_data[i] = float(phone_data[i])

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
        """
        This function is used to generate the prediction from the real-time data collected
        :param data: a window of raw data
        """
        if self.is_classifying is True:
            if self.method_text.get() == 'tsfresh':
                data_to_predict = data.values
            elif self.method_text.get() == 'Time/Frequency':
                data_to_predict = [data]

            probability = self.clf6.predict_proba(data_to_predict)
            prediction_prob = {}
            for i in range(0, len(probability[0])):
                prediction_prob[self.clf6.classes_[i]] = probability[0][i]
            sorted_prob = sorted(prediction_prob.items(), key=lambda kv: kv[1], reverse=True)
            self.update_window(sorted_prob)

    def classify_btn_hit(self):
        """
        This function is triggered by the classify button press, it generate a
        new window and start performing real-time classification on smartphone activities
        """
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())

        if self.method_text.get() == 'tsfresh':
            target_file.columns = ['index', 'target']
        elif self.method_text.get() == 'Time/Frequency':
            target_file.columns = ['target']
        self.change_read_to_scroll(target_file)

        self.clf6.fit(data_file.values, target_file['target'])
        self.is_classifying = True

        self.window.deiconify()

    def select_classify_method(self):
        """
        toggle the text appeared for the button, the method selected should match with
        the method used for feature extraction
        """
        if self.method_text.get() == "Time/Frequency":
            self.method_text.set("tsfresh")
        elif self.method_text.get() == "tsfresh":
            self.method_text.set("Time/Frequency")

    def select_label_combination(self):
        """
        toggle the text appeared for the button, the method selects whether the combination of
        reading and scrolling activities should be combined or not
        """
        if self.combine_text.get() == "Combine reading & scrolling":
            self.combine_text.set("Do not combine")
        elif self.combine_text.get() == "Do not combine":
            self.combine_text.set("Combine reading & scrolling")

    def validate_btn_hit(self):
        """
        This function is trigger by the validate button press, it would read the files been
        browsed and analyse the performance of the ML models on the data given
        """
        data_file = pd.read_csv(self.data_file_path.get(), encoding='utf-8')
        target_file = pd.read_csv(self.target_file_path.get(), encoding='utf-8')

        if self.method_text.get() == 'tsfresh':
            target_file.columns = ['index', 'target']
        elif self.method_text.get() == 'Time/Frequency':
            target_file.columns = ['target']

        target_file = self.change_read_to_scroll(target_file)

        self.output_ml_validate_score(data_file.values, target_file['target'])

    def output_ml_validate_score(self, data, target):
        """
        perform 5 folds cross validation for the ML model trained
        """

        scaler = StandardScaler()
        scaler.fit(data)
        data2 = scaler.transform(data)

        self.clf.fit(data2, target)
        scores = cross_val_score(self.clf, data2, target, cv=5)
        print("MLP: " + str(scores.mean()))

        self.clf2.fit(data, target)
        scores = cross_val_score(self.clf2, data, target, cv=5)
        print("RF: " + str(scores.mean()))

        self.clf3.fit(data, target)
        scores = cross_val_score(self.clf3, data, target, cv=5)
        print("KNN: " + str(scores.mean()))

        self.clf4.fit(data, target)
        scores = cross_val_score(self.clf4, data, target, cv=5)
        print("SVM: " + str(scores.mean()))

        self.clf5.fit(data, target)
        scores = cross_val_score(self.clf5, data, target, cv=5)
        print("Bagging: " + str(scores.mean()))

        self.clf6.fit(data, target)
        scores = cross_val_score(self.clf6, data, target, cv=5)
        print("ExtraTree: " + str(scores.mean()))

        self.clf7.fit(data, target)
        scores = cross_val_score(self.clf7, data, target, cv=5)
        print("AdaBoosting: " + str(scores.mean()))

    def confusion_matrix(self):
        """
        This function generated the confusion matrix for the training set provided
        by the files user browsed
        :return: a new window with generated confusion matrix
        """
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())

        target_file.columns = ['target']
        target_file = self.change_read_to_scroll(target_file)

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(data_file, target_file, random_state=0)
        y_pred = self.clf6.fit(X_train, y_train).predict(X_test)

        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, target_file['target'], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

    def validate_by_random(self):
        """
        This function validates the algorithms for the ML model by
        randomising the labels in the target file
        """
        data_file = pd.read_csv(self.data_file_path.get())
        target_file = pd.read_csv(self.target_file_path.get())

        if self.method_text.get() == 'tsfresh':
            target_file.columns = ['index', 'target']
        elif self.method_text.get() == 'Time/Frequency':
            target_file.columns = ['target']

        self.change_read_to_scroll(target_file)

        lables = unique_labels(target_file['target'])

        output_target = []
        for i in range(0, len(data_file.values)):
            output_target.append(lables[random.randrange(len(lables))])

        target_df = pd.DataFrame(output_target)
        target_df.columns = ['target']

        self.output_ml_validate_score(data_file, target_df['target'])

    def change_read_to_scroll(self, targetData):
        """
        This function combines the label of reading and scrolling
        into one label, which is scrolling
        :param targetData: the target file of a training set
        :return: a file with label combined
        """
        if self.combine_text == "Combine reading & scrolling":
            targetData.loc[targetData['target'] == 'Sitting_Read', 'target'] = 'Sitting_Scroll'
            targetData.loc[targetData['target'] == 'Walking_Read', 'target'] = 'Walking_Scroll'
            targetData.loc[targetData['target'] == 'Multitasking_Read', 'target'] = 'Multitasking_Scroll'

        return targetData

    def plot_learning_curve(self):
        data_file = pd.read_csv(self.data_file_path.get())
        labels = []
        i=0
        while i<71:
            labels.append("data {}".format(i))
            i += 1

        data_file.columns = labels
        target_file = pd.read_csv(self.target_file_path.get())
        data_file.insert(loc=71, column='label', value=target_file)
        #print(data_file.info())

        train_sizes, train_scores, test_scores = learning_curve(estimator=self.clf6,
                                                                X=data_file[labels],
                                                                y=np.ravel(data_file['label']),
                                                                train_sizes = [10, 50, 250, 1000, 2000, 4000],
                                                                cv=5,
                                                                shuffle=True
                                                                )
        #train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        values = pd.Series(test_scores_mean, index=train_sizes)
        #print('Mean training scores\n\n', pd.Series(train_scores_mean, index=train_sizes))
        print('\n', '-' * 20)  # separator
        print('\nMean validation scores\n\n', pd.Series(test_scores_mean, index=train_sizes))

        plot = values.plot(kind='line')
        plot.set_xlabel("train_size")
        plot.set_ylabel("accuracy")
        plot.set_title("learning_curve")

        plot.plot
        plt.show()











def browse_btn_hit(folder_path):
    """
    This function used to trigger the browse button event to save the
    file path to the inout parameter
    :param folder_path: the StringVar that stores the file path
    """
    filename = filedialog.askopenfilename()
    folder_path.set(filename)
    print(filename)






def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(classes)
    lables = []
    for lable in classes:
        lable = lable.replace("_", " ")
        lable = lable.replace("Multitasking","Cognitive Load")
        lables.append(lable)
    lables = np.array(lables)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=lables, yticklabels=lables,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
