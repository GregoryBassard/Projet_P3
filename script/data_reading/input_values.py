from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        print("Client initialisé")

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time):
        if _time >= 0:
            print(f'Time: {_time} ms')
            state = iface.get_simulation_state()
            print(f'x: {round(state.position[0])}, y: {round(state.position[1])}, z: {round(state.position[2])}')
            # print(f'gear: {state.scene_mobil.gearbox_state}, speed: {state.display_speed} km/h')
        else:
            print(f'Time: {_time} ms')



            


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    
    try:
        print("Tentative de connexion...")
        run_client(MainClient(), server_name)
    except Exception as e:
        print(f"Erreur lors de la connexion: {e}")
        print("Vérifiez que TMInterface est lancé et qu'une instance est disponible")


if __name__ == '__main__':
    main()
