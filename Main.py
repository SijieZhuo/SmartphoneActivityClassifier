import os
import tkinter as tk
from tkinter import font as tkfont
import StartPage
import CollectPage
import FeatureExtractionPage
import ClassifyPage

# sampling frequency is 50 hz
window_size = 250


class ClassifierApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        if not os.path.exists("Data"):
            os.makedirs("Data")

        if not os.path.exists("Data/DataForAnalysation"):
            os.makedirs("Data/DataForAnalysation")

        if not os.path.exists("Data/TrainingSet"):
            os.makedirs("Data/TrainingSet")

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.winfo_toplevel().title("Smartphone activity classification")
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.current_data = PhoneData()
        self.current_data.data = []

        self.frames = {}
        for F in (StartPage.StartPage, CollectPage.CollectPage, FeatureExtractionPage.FeatureExtractionPage,
                  ClassifyPage.ClassifyPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class PhoneData(object):
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


if __name__ == "__main__":
    main = ClassifierApp()
    main.wm_geometry("550x400")
    main.title("Smartphone activity classification")
    main.mainloop()
