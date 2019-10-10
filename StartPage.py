import _thread
import os
import tkinter as tk
import bluetooth


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        connect_btn = tk.Button(self, text="Connect to the device", command=lambda: bt_connect_hit(self), width=25)
        connect_btn.grid(row=1, column=1, pady=10)

        self.collect_page_btn = tk.Button(self, text="Collect data", state=tk.DISABLED,
                                          command=lambda: controller.show_frame("CollectPage"), width=25)
        self.collect_page_btn.grid(row=2, column=1, pady=10)

        self.feature_page_btn = tk.Button(self, text="Feature extraction",
                                          command=lambda: controller.show_frame("FeatureExtractionPage"), width=25)
        self.feature_page_btn.grid(row=3, column=1, pady=10)

        self.classify_page_btn = tk.Button(self, text="Classify",
                                           command=lambda: controller.show_frame("ClassifyPage"), width=25)
        self.classify_page_btn.grid(row=4, column=1, pady=10)

        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_rowconfigure((0,5), weight=1)

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


def collect_data(client, main):
    try:
        while True:
            data = client.recv(1024)
            if len(data) == 0: break
            main.current_data.data = data
    except IOError:
        pass
