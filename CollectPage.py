import csv
import datetime
import os
import re
import tkinter as tk


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

        back_btn = tk.Button(self, text="back", command=lambda: controller.show_frame("StartPage"))
        back_btn.grid(row=6, column=1)

        self.folderName = ""
        self.currentActivity = "finished"
        self.isRecording = False

    def update_data(self, data):

        data2 = data.decode("utf-8")
        data3 = data2.replace(' ', '')
        data4 = data3.replace('[', '')
        phone_data = data4.replace(']', '').split(',')

        if len(phone_data) != 50:
            print("current: " + self.currentActivity)
            print(phone_data)

            if len(phone_data) == 1:
                # print(phone_data)
                if phone_data[0].count(phone_data[0][:5]) > 1:
                    phone_data[0] = phone_data[0][:int((len(phone_data[0]) / 2))]
                    print(phone_data[0])
                self.update_avtivity(phone_data[0])
            elif len(phone_data) == 2:
                print(phone_data)
                csvName = "Data/" + self.folderName + "/" + self.folderName + "_questions.csv"
                print(csvName)
                with open(csvName, 'a', newline='') as writeFile:
                    writer = csv.writer(writeFile)
                    writer.writerows(phone_data)
                    print("write")
                writeFile.close()
        else:
            if self.isRecording is True:
                print(phone_data[0] + "  " + phone_data[-1])

                string = re.findall('[a-zA-Z_]', phone_data[0])
                front = ''.join(string)
                string2 = re.findall('[a-zA-Z_]', phone_data[-1])
                rear = ''.join(string2)
                if front != '':
                    self.update_avtivity(front)
                    return
                if rear != '':
                    self.update_avtivity(rear)
                    return

                phone_data[0] = re.sub('[a-zA-Z_]', '', phone_data[0])
                phone_data[-1] = re.sub('[a-zA-Z_]', '', phone_data[-1])

                for i in range(0,len(phone_data)):
                    if i%10 != 0:
                        phone_data[i] = float(phone_data[i])
                        #phone_data[i].astype(float32)

                #float_data = [float(i) for i in phone_data]
                separated = [phone_data[x:x + 10] for x in range(0, len(phone_data), 10)]

                for row in separated:
                    row.append(self.currentActivity)

                fileName = "Data/" + self.folderName + "/" + self.folderName + "_" + self.currentActivity + ".csv"
                with open(fileName, 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    for row in separated:
                        writer.writerow(row)
                csvFile.close()

    def update_avtivity(self, activity):
        if self.currentActivity == "finished":
            if activity != "finished":
                self.setup_activity_folder(activity)
                self.isRecording = True
        else:  # current activity is not finished
            if activity == "finished":
                self.change_label_color("green")
                self.isRecording = False
                self.currentActivity = "finished"

    def setup_activity_folder(self, name):
        self.currentActivity = name
        self.change_label_color("blue")
        csvName = "Data/" + self.folderName + "/" + self.folderName + "_" + name + ".csv"
        print(csvName)
        with open(csvName, 'w', newline='') as writeFile:
            writer = csv.writer(writeFile)
            line = [["time","accX", "accY", "accZ", "rotX", "rotY", "rotZ", "graX", "graY", "graZ", "activity"]]
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
