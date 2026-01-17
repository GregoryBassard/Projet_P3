from tminterface.interface import TMInterface
from tminterface.client import Client
from multiprocessing import Process, Pipe
from script.defs.defs import server_connection_process
from script.tm_class.tm_classes import TMInfos, TMInterfaceClient

import time

SERVER_NAME = 'TMInterface0'

if __name__ == '__main__':
    client = TMInterfaceClient()
    iface = TMInterface(SERVER_NAME)
    
    print(f"Connecting to server: {SERVER_NAME}")
    conn1, conn2 = Pipe()
    client_process = Process(target=server_connection_process, args=(conn1, iface, client))
    client_process.start()
    while True:
        time.sleep(1)
        conn2.send(0)
        print("Main process alive")