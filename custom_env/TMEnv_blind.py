import logging
from gymnasium import Env, spaces
from multiprocessing import Process, Pipe
from client.client import server_connection_process
from tminterface.interface import TMInterface
import time
import numpy as np
from script.tm_classes.tm_classes import TMInterfaceClient

class TrackmaniaEnv(Env):
    def __init__(self, speed=1, server_name='TMInterface0', map_name='challenge_1', max_race_time=20.0, mode='normal'):
        # server connection
        self.map_name = map_name

        self.conn1, self.conn2 = Pipe(duplex=True)

        self.speed = speed
        self.server_name = server_name

        self.mode = mode

        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(11,), dtype=float)

        # Placeholder start position
        self.position = (0, 0, 0)

        # Set max race time
        self.max_race_time = max_race_time # seconds

        self.stuck_time_start = None
        self.stuck_time = 3 # seconds

        # init flag
        self.init = False

        self.starting_pos = 0.0

        self.off_track_threshold_y = 15.0

        self.best_score = self.max_race_time

        self.start_time = 0.0
        self.nb_actions = 0

        self.score = 0.0

        self.finished = False

        self.logger = logging.getLogger('__main__')
        self.logger.info("Env Started")

        self.cur_cp_count = 0


    def connect_to_server(self):
        print(f'Connecting to {self.server_name}...')
        try:
            client_process = Process(target=server_connection_process, args=(self.conn2, TMInterface(self.server_name), TMInterfaceClient(_speed_IG=self.speed, _map_path=self.map_name), True))
            client_process.start()
            confirmation = self.conn1.recv()
            if confirmation != "ready":
                print("Error while connecting to server")
            else:
                print(f"Connected to server. {confirmation}")
        except Exception as e:
            print(f"Connection error: {e}")

    def server_get_info(self):
        self.conn1.send(-2) 
        info = self.conn1.recv()
        if info is None:
            self.logger.error("No info received from server.")
            info = {
                'pos': (0.0, 0.0, 0.0),
                'vel': (0.0, 0.0, 0.0),
                'ypr': (0.0, 0.0, 0.0),
                'speed': 0.0,
                'turning_rate': 0.0,
                'cur_cp_count': 0,
            }
        return info
    
    def server_request_respawn(self):
        self.conn1.send(-1)

    def step(self, action):
        if self.nb_actions == 0:
            self.start_time = time.time()
        self.nb_actions += 1

        # Apply action
        # Send input to server
        self.conn1.send(action)

        # Get new info from server
        self.info = self.server_get_info()
        self.position = self.info['pos']
        new_cur_cp_count = self.info['cur_cp_count']

        # Compute reward
        if self.mode is 'normal':
            reward, done = self.reward_normal()
        elif self.mode is 'ck':
            reward, done = self.reward_ck(new_cur_cp_count)
        
        if done is False:
            # Check if car is stuck
            reward, done = self.car_stuck()

        obs = self._get_obs(self.info)

        return obs, reward, done, False, {}
    
    def reward_normal(self):
        reward = 0.0
        done = False

        # Check if race is done
        if self.info['race_time'] / 1000 >= self.max_race_time:
            reward = -100.0
            self.score = self.max_race_time
            done = True
        elif self.info['race_finished']:
            reward = 100.0 - (self.info['race_time'] / 1000)
            self.score = self.info['race_time'] / 1000
            self.logger.info("Race finished in %.2f seconds", self.score)
            self.finished = True
            done = True

        return reward, done
    
    def reward_ck(self, new_cur_cp_count):
        reward = 0.0
        done = False

        # Check if race is done
        if self.info['race_time'] / 1000 >= self.max_race_time:
            reward = -100.0
            self.score = self.max_race_time
            done = True
        elif self.info['race_finished']:
            reward = (100.0 * (self.cur_cp_count + 1)) - (self.info['race_time'] / 1000)
            self.score = self.info['race_time'] / 1000
            self.logger.info("Race finished in %.2f seconds with reward %.2f", self.score, reward)
            self.finished = True
            done = True
        elif self.cur_cp_count < new_cur_cp_count: # Check if new checkpoint reached
            reward = (100.0 * (new_cur_cp_count)) - (self.info['race_time'] / 1000)
            self.cur_cp_count = new_cur_cp_count
            self.logger.info("Checkpoint %d reached. Reward: %.2f", self.cur_cp_count, reward)
        
        return reward, done

    def car_stuck(self):
        done = False
        reward = 0.0
        if self.info['speed'] < 10:
            if self.stuck_time_start is None:
                self.stuck_time_start = self.info['race_time']
            else:
                if (self.info['race_time'] - self.stuck_time_start) / 1000 >= self.stuck_time:
                    self.logger.debug("Car stuck. Ending episode.")
                    reward = -100.0
                    self.score = self.max_race_time
                    done = True
        else:
            self.stuck_time_start = None
        return reward, done

    def render(self):
        pass

    def reset(self):
        if self.init is False:
            self.connect_to_server()
            # self.init = True
            print("Waiting 10 sec")
            time.sleep(10)

        self.server_request_respawn()
        time.sleep(1)

        info = self.server_get_info()

        obs = self._get_obs(info)
        self.position = info['pos']

        if self.init is False:
            self.starting_pos = round(self.position[2], 1)
            self.init = True

        self.start_time = 0.0
        self.nb_actions = 0
        self.finished = False
        self.cur_cp_count = 0
        self.score = 0.0

        return obs

    def _get_obs(self, info: dict) -> np.ndarray:
        info_clean = {
            'pos': np.array(info['pos'], dtype=float),
            'vel': np.array(info['vel'], dtype=float),
            'pitch': np.array([info['ypr'][1]], dtype=float),
            'roll': np.array([info['ypr'][2]], dtype=float),
            'speed': np.array([info['speed']], dtype=float),
            'turning_rate': np.array([info['turning_rate']], dtype=float),
            'cur_cp_count': np.array([info['cur_cp_count']], dtype=float)
        }
        return np.concatenate([
            info_clean["speed"].flatten(),
            info_clean["vel"].flatten(),
            info_clean["turning_rate"].flatten(),
            info_clean["pos"].flatten(),
            info_clean["pitch"].flatten(),
            info_clean["roll"].flatten(),
            info_clean["cur_cp_count"].flatten()
        ])

    def close(self):
        if self.init:
            self.conn1.send(99) # close signal
            self.init = False