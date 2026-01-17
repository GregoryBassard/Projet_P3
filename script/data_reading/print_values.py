from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            # print(f'Time: {_time} ms')
            state = iface.get_simulation_state()
            # print(f'x: {round(state.position[0])}, y: {round(state.position[1])}, z: {round(state.position[2])}')
            print(f'gear: {state.scene_mobil.engine.gear}, rear_gear: {state.scene_mobil.engine.rear_gear}')

        # if _time == 1000:
        #     state = iface.get_simulation_state()
        #     print(state)

        # if _time >= 2000:
        #     state = iface.get_simulation_state()

        #     wheels = state.simulation_wheels
        #     ground_contact = [wheel.real_time_state.has_ground_contact for wheel in wheels]
        #     print(f'Wheels sliding: {ground_contact}')

def main():
    server_name = f'TMInterface1'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()
