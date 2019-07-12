import csv
import tkinter as tk
import StartPage
import datetime
import numpy as np
import os
import re


class CollectPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # print(self.controller.frames)
        self.data = controller.current_data
        self.data.bind_to(self.update_data)
        # print(self.data)

        self.walk_idle = tk.Label(self, text="Walk_Idle")
        self.walk_idle.grid(row=0, column=0)
        self.walk_type = tk.Label(self, text="Walk_Type")
        self.walk_type.grid(row=1, column=0)
        self.walk_read = tk.Label(self, text="Walk_Read")
        self.walk_read.grid(row=2, column=0)
        self.walk_scroll = tk.Label(self, text="Walk_Scroll")
        self.walk_scroll.grid(row=3, column=0)
        self.walk_watch = tk.Label(self, text="Walk_Watch")
        self.walk_watch.grid(row=4, column=0)

        self.sit_idle = tk.Label(self, text="Sit_Idle")
        self.sit_idle.grid(row=0, column=1)
        self.sit_type = tk.Label(self, text="Sit_Type")
        self.sit_type.grid(row=1, column=1)
        self.sit_read = tk.Label(self, text="Sit_Read")
        self.sit_read.grid(row=2, column=1)
        self.sit_scroll = tk.Label(self, text="Sit_Scroll")
        self.sit_scroll.grid(row=3, column=1)
        self.sit_watch = tk.Label(self, text="Sit_Watch")
        self.sit_watch.grid(row=4, column=1)

        self.multi_type = tk.Label(self, text="Multi_Type")
        self.multi_type.grid(row=1, column=2)
        self.multi_read = tk.Label(self, text="Multi_Read")
        self.multi_read.grid(row=2, column=2)
        self.multi_scroll = tk.Label(self, text="Multi_Scroll")
        self.multi_scroll.grid(row=3, column=2)
        self.multi_watch = tk.Label(self, text="Multi_Watch")
        self.multi_watch.grid(row=4, column=2)

        self.record_btn_text = tk.StringVar()
        record_btn = tk.Button(self, textvariable=self.record_btn_text, command=lambda: record_btn_hit(self))
        self.record_btn_text.set("start recording")
        record_btn.grid(row=5, column=1)

        self.folderName = ""
        self.currentActivity = ""
        self.isRecording = False

    def update_data(self, data):

        data2 = data.decode("utf-8")
        data3 = data2.replace(' ', '')
        data4 = data3.replace('[', '')
        phone_data = data4.replace(']', '').split(',')

        if len(phone_data) != 45:
            print(self.folderName)
            if len(phone_data) == 1:
                if phone_data[0] != "finished":
                    self.setup_activity_folder(phone_data[0])
                    self.isRecording = True
                else:
                    self.change_label_color("green")
                    self.isRecording = False
        else:
            if self.isRecording is True:

                float_data = [float(i) for i in phone_data]
                separated = [float_data[x:x + 9] for x in range(0, len(float_data), 9)]
                for row in separated:
                    row.append(self.currentActivity)

                fileName = "Data/" + self.folderName + "/" + self.currentActivity + ".csv"
                with open(fileName, 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    for row in separated:
                        writer.writerow(row)
                csvFile.close()

    def setup_activity_folder(self, name):
        self.currentActivity = name
        self.change_label_color("blue")
        csvName = "Data/" + self.folderName + "/" + name + ".csv"
        print(csvName)
        with open(csvName, 'w', newline='') as writeFile:
            writer = csv.writer(writeFile)
            line = [["accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]]
            writer.writerows(line)
            print("write")
        writeFile.close()

    def change_label_color(self, colour):
        if self.currentActivity == "Walking_Type":
            self.walk_type.configure(foreground=colour)
        elif self.currentActivity == "Walking_Read":
            self.walk_read.configure(foreground=colour)
        elif self.currentActivity == "Walking_Watch":
            self.walk_watch.configure(foreground=colour)
        elif self.currentActivity == "Walking_Scroll":
            self.walk_scroll.configure(foreground=colour)
        elif self.currentActivity == "Walking_Idle":
            self.walk_idle.configure(foreground=colour)
        elif self.currentActivity == "Sitting_Type":
            self.sit_type.configure(foreground=colour)
        elif self.currentActivity == "Sitting_Read":
            self.sit_read.configure(foreground=colour)
        elif self.currentActivity == "Sitting_Watch":
            self.sit_watch.configure(foreground=colour)
        elif self.currentActivity == "Sitting_Scroll":
            self.sit_scroll.configure(foreground=colour)
        elif self.currentActivity == "Sitting_Idle":
            self.sit_idle.configure(foreground=colour)
        elif self.currentActivity == "Multitasking_Type":
            self.multi_type.configure(foreground=colour)
        elif self.currentActivity == "Multitasking_Read":
            self.multi_read.configure(foreground=colour)
        elif self.currentActivity == "Multitasking_Watch":
            self.multi_watch.configure(foreground=colour)
        elif self.currentActivity == "Multitasking_Scroll":
            self.multi_scroll.configure(foreground=colour)


def record_btn_hit(page):
    time = datetime.datetime.now()
    page.folderName = time.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs("Data/" + page.folderName)
    page.record_btn_text.set("recording")
