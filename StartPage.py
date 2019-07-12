import tkinter as tk
from typing import List, Any
import os
import datetime

import Main
import bluetooth
import _thread
import CollectPage


class StartPage(tk.Frame):
    # list that stores the available devices
    devices: List[Any] = []

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        connect_btn = tk.Button(self, text="Connect to the device", command=lambda: bt_connect_hit(self))
        connect_btn.pack()

        self.collect_page_btn = tk.Button(self, text="Collect data", state=tk.DISABLED,
                                          command=lambda:collect_btn_hit(self))
        self.collect_page_btn.pack()

    # refresh the bt devices options in the dropdown menu
    def refresh_option(self):
        self.variable.set('')
        self.dropdown['menu'].delete(0, 'end')

        for device in self.devices:
            self.dropdown['menu'].add_command(label=device[0], command=tk._setit(self.variable, device[0]))

        self.variable.set(self.devices[0][0])


'''=========================== function for the bt connect button =============================='''


def bt_connect_hit(page):
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    uuid = "6bfc8497-b445-406e-b639-a5abaf4d9739"

    bluetooth.advertise_service(server_sock, "SampleServer",
                                service_id=uuid,
                                service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                                profiles=[bluetooth.SERIAL_PORT_PROFILE],
                                #                   protocols = [ OBEX_UUID ]
                                )

    print("Waiting for connection on RFCOMM channel %d" % port)

    client_sock, client_info = server_sock.accept()
    print("Accepted connection from ", client_info)
    page.collect_page_btn['state'] = 'normal'

    _thread.start_new_thread(collect_data, (client_sock, page.controller))


def collect_btn_hit(page):
    page.controller.show_frame("CollectPage")
    if not os.path.exists("Data"):
        os.makedirs("Data")





def collect_data(client, main):
    try:
        while True:
            data = client.recv(1024)
            if len(data) == 0: break
            main.current_data.data = data
    except IOError:
        pass
