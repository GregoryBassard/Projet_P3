from tminterface.interface import TMInterface
from tminterface.client import Client

class TMInfos:
    def __init__(self):
        self.speed = 0
        self.position = (0, 0, 0)
        self.velocity = (0, 0, 0)
        self.ypr = (0, 0, 0)
        self.turning_rate = 0
        # self.input_steer = 0
        self.input_left = False
        self.input_right = False
        self.input_accelerate = False
        self.input_brake = False
        self.race_time = 0
        self.race_finished = False
        self.cur_cp_count = 0
    
    def update(self, data):
        self.speed = data['speed']
        self.position = (data['pos'][0], data['pos'][1], data['pos'][2])
        self.velocity = (data['vel'][0], data['vel'][1], data['vel'][2])
        self.ypr = (data['ypr'][0], data['ypr'][1], data['ypr'][2])
        self.turning_rate = data['turning_rate']
        # self.input_steer = data['input_steer']
        self.input_left = data['input_steer']
        self.input_right = data['input_steer']
        self.input_accelerate = data['input_accelerate']
        self.input_brake = data['input_brake']
        self.race_time = data['race_time']
        self.race_finished = data['race_finished']
        self.cur_cp_count = data['cur_cp_count']

    def get_info_blind(self, state) -> dict:
        data = {}

        # Position and orientation
        data['pos'] = state.position  # [x, y, z] coordinates
        data['vel'] = state.velocity  # [vx, vy, vz] velocity vector
        data['ypr'] = state.yaw_pitch_roll  # [yaw, pitch, roll] angles

        # Speed calculations
        data['speed'] = state.display_speed  # Speed in km/h as displayed in game

        # Vehicle state
        data['turning_rate'] = state.scene_mobil.turning_rate  # Turning rate

        # Input states
        # data['input_steer'] = state.input_steer  # Steering input (-65536 to 65536)
        data['input_left'] = state.input_left  # Left button (bool)
        data['input_right'] = state.input_right  # Right button (bool)
        data['input_accelerate'] = state.input_accelerate  # Accelerate button (bool)
        data['input_brake'] = state.input_brake  # Brake button (bool)

        # Time information
        data['race_time'] = state.race_time  # Current race time in milliseconds
        data['race_finished'] = state.player_info.race_finished  # Race finished status

        # Checkpoint information
        data['cur_cp_count'] = state.player_info.cur_cp_count  # Current checkpoint index

        return data
    
    def get_info_road(self, state) -> dict:
        data = {}

        data['gear'] = state.scene_mobil.engine.gear  # Current gear /1

        wheels = state.simulation_wheels
        data['is_sliding'] = [wheel.real_time_state.is_sliding for wheel in wheels] # /4
        data['has_ground_contact'] = [wheel.real_time_state.has_ground_contact for wheel in wheels] # /4

        # Position and orientation
        data['pos'] = state.position  # [x, y, z] coordinates /3
        data['vel'] = state.velocity  # [vx, vy, vz] velocity vector /3
        data['ypr'] = state.yaw_pitch_roll  # [yaw, pitch, roll] angles /3

        # Speed calculations
        data['speed'] = state.display_speed  # Speed in km/h /1

        # Vehicle state
        data['turning_rate'] = state.scene_mobil.turning_rate  # Turning rate /1

        # Time information
        data['race_time'] = state.race_time  # Current race time in milliseconds
        data['race_finished'] = state.player_info.race_finished  # Race finished status
        return data       
    
    def return_data_dict(self):
        return {
            'speed': self.speed,
            'position': self.position,
            'velocity': self.velocity,
            'ypr': self.ypr,
            'turning_rate': self.turning_rate,
            # 'input_steer': self.input_steer,
            'input_left': self.input_left,
            'input_right': self.input_right,
            'input_accelerate': self.input_accelerate,
            'input_brake': self.input_brake,
            'cur_cp_count': self.cur_cp_count
        }
    
    def __str__(self):
        # return f"data IG : speed={self.speed}, position={self.position}, velocity={self.velocity}, ypr={self.ypr}, turning_rate={self.turning_rate}, input_steer={self.input_steer}, input_accelerate={self.input_accelerate}, input_brake={self.input_brake}, race_time={self.race_time}, race_finished={self.race_finished}"
        return f"data IG : speed={self.speed}, position={self.position}, velocity={self.velocity}, ypr={self.ypr}, turning_rate={self.turning_rate}, input_left={self.input_left}, input_right={self.input_right}, input_accelerate={self.input_accelerate}, input_brake={self.input_brake}, race_time={self.race_time}, race_finished={self.race_finished}, cur_cp_count={self.cur_cp_count}"
    
class TMInterfaceClient(Client):
    def __init__(self, _speed_IG: float = 1, _map_path: str = "test2") -> None:
        self.speed_IG = _speed_IG
        self.map_path = _map_path
        super(TMInterfaceClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        iface.execute_command(f"map \"My Challenges\\{self.map_path}.Challenge.Gbx\"")
        iface.set_speed(self.speed_IG)
        iface.set_timeout(1000) # 1 sec

    def on_custom_command(self, iface: TMInterface, time_from: int, time_to: int, command: str, args: list) -> None:
        pass

    def on_run_step(self, iface: TMInterface, _time) -> None:
        pass