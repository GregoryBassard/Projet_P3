import logging
from gymnasium import Env, spaces
from multiprocessing import Process, Pipe
from client.client import server_connection_process
from tminterface.interface import TMInterface
import time
import numpy as np
from script.tm_classes.tm_classes import TMInterfaceClient

class TrackmaniaEnv(Env):
    def __init__(self, speed, server_name, map_name, max_race_time, road_pos, off_track_threshold_y=40.0):
        # server connection
        self.map_name = map_name

        self.conn1, self.conn2 = Pipe(duplex=True)

        self.speed = speed
        self.server_name = server_name
        self.road_pos = road_pos

        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,), dtype=float)
        # Placeholder start position

        # Set max race time
        self.max_race_time = max_race_time # seconds

        self.stuck_time_start = None
        self.stuck_time = 3 # seconds

        # init flag
        self.init = False

        self.off_track_threshold_y = off_track_threshold_y

        self.best_score = self.max_race_time

        self.start_time = 0.0

        self.score = 0.0
        self.reward = 0.0

        self.angle_diff = 0.0
        self.distance_to_road = 0.0

        self.race_finished = False

        self.logger = logging.getLogger('__main__')
        self.logger.info("Env Started")

        self.progression = 0.0

    def connect_to_server(self):
        self.logger.info(f'Connecting to {self.server_name}...')
        try:
            client_process = Process(target=server_connection_process, args=(self.conn2, TMInterface(self.server_name), TMInterfaceClient(_speed_IG=self.speed, _map_path=self.map_name), False))
            client_process.start()
            confirmation = self.conn1.recv()
            if confirmation != "ready":
                self.logger.error("Error while connecting to server")
            else:
                self.logger.info(f"Connected to server. {confirmation}")
        except Exception as e:
            self.logger.error(f"Connection error: {e}")

    def server_get_info(self):
        self.conn1.send(-2)
        info = self.conn1.recv()
        if type(info) is not dict:
            self.logger.error(f"No info received from server: {info}")
            info = {
                'pos': (0.0, 0.0, 0.0),
                'vel': (0.0, 0.0, 0.0),
                'ypr': (0.0, 0.0, 0.0),
                'speed': 0.0,
                'turning_rate': 0.0,
                'gear': 0,
                'is_sliding': (False, False, False, False),
                'has_ground_contact': (True, True, True, True),
                'race_finished': False,
                'race_time': 0.0
            }
        return info

    def server_request_respawn(self):
        self.conn1.send(-1)

    def step(self, action):
        # Apply action
        # Send input to server
        self.conn1.send(action)

        # Get new info from server
        self.info = self.server_get_info()
        self.angle_diff, self.distance_to_road = self.get_road_angle_and_distance(self.info['pos'], self.road_pos)

        # # Compute reward
        self.reward += self.calculate_reward()
        self.reward += -0.1

        # check if race is done
        done = False
        if self.info['race_finished']:
            done = True
            race_time = self.info['race_time'] / 1000.0

            self.score = race_time
            self.race_finished = True

        if done is False:
            # Check if race time exceeded
            if self.info['race_time'] / 1000 >= self.max_race_time:
                self.score = self.max_race_time
                done = True

            if done is False:
                # Check if car is stuck
                reward, done = self.car_stuck()
                self.reward += reward
            if done is False:
                # Check if off track
                if self.info['pos'][1] < self.off_track_threshold_y:
                    self.logger.debug("Car off track. Ending episode.")
                    self.reward += -10.0
                    done = True
                    self.score = self.max_race_time

        obs = self._get_obs(self.info)

        return obs, self.reward, done, False, {}

    def get_closest_index(self, car_pos, road_pos):
        road_pos_array = np.array(road_pos)

        # Find the closest point on the road to the car
        road_x, road_y, road_z = road_pos_array[:, 0], road_pos_array[:, 1], road_pos_array[:, 2]
        car_x, car_y, car_z = car_pos[0], car_pos[1], car_pos[2]

        distances = np.sqrt((road_x - car_x) ** 2 + (road_y - car_y) ** 2 + (road_z - car_z) ** 2)
        closest_index = np.argmin(distances)

        return closest_index, distances, len(road_pos)

    def get_road_angle_and_distance(self, car_pos, road_pos):
        # Find the closest point on the road to the car
        closest_index, distances, _ = self.get_closest_index(car_pos, road_pos)

        # Calculate distance to the road
        distance = distances[closest_index]

        # Calculate angle between car's heading and road direction
        if closest_index < len(road_pos) - 1:
            next_point = road_pos[closest_index + 1]
        else:
            next_point = road_pos[closest_index - 1]

        road_direction = np.arctan2(next_point[1] - road_pos[closest_index][1], next_point[0] - road_pos[closest_index][0])
        car_direction = np.arctan2(self.info['vel'][1], self.info['vel'][0])

        angle_diff = road_direction - car_direction
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize

        return angle_diff, distance

    def calculate_reward(self):
        closest_index, _, total_index = self.get_closest_index(self.info['pos'], self.road_pos)
        new_progression = closest_index / total_index

        reward = (new_progression - self.progression) * 100.0 
        if reward != 0.0:
            self.logger.debug(f"(new_progression - old_progression)= ({new_progression:.3f} - {self.progression:.3f}) = {reward:.3f}")

        self.progression = new_progression

        return reward

    def car_stuck(self):
        done = False
        reward = 0.0
        if self.info['speed'] < 10:
            if self.stuck_time_start is None:
                self.stuck_time_start = self.info['race_time']
            else:
                if (self.info['race_time'] - self.stuck_time_start) / 1000 >= self.stuck_time:
                    self.logger.debug("Car stuck. Ending episode.")
                    reward = -10.0
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
            self.logger.debug("Waiting 5 sec")
            time.sleep(5)

        self.server_request_respawn()
        time.sleep(0.1)

        info = self.server_get_info()
        self.logger.debug(f"info: {info}")
        obs = self._get_obs(info)
        self.logger.debug(f"obs: {obs}")

        if self.init is False:
            self.init = True

        self.start_time = 0.0
        self.race_finished = False
        self.stuck_time_start = None

        self.score = 0.0
        self.reward = 0.0
        self.progression = 0.0

        self.angle_diff = 0.0
        self.distance_to_road = 0.0

        return obs

    def _get_obs(self, info: dict) -> np.ndarray:
        info_clean = {
            'vel': np.array(info['vel'], dtype=float),
            'yaw': np.array([info['ypr'][0]], dtype=float),
            'pitch': np.array([info['ypr'][1]], dtype=float),
            'roll': np.array([info['ypr'][2]], dtype=float),
            'speed': np.array([info['speed']], dtype=float),
            'turning_rate': np.array([info['turning_rate']], dtype=float),
            'gear': np.array([info['gear']], dtype=int),
            'is_sliding': np.array(info['is_sliding'], dtype=bool),
            'has_ground_contact': np.array(info['has_ground_contact'], dtype=bool),
            'angle_to_road': np.array([self.angle_diff], dtype=float),
            'distance_to_road': np.array([self.distance_to_road], dtype=float)
        }
        return np.concatenate([
            info_clean["vel"].flatten(),
            info_clean["yaw"].flatten(),
            info_clean["pitch"].flatten(),
            info_clean["roll"].flatten(),
            info_clean["speed"].flatten(),
            info_clean["turning_rate"].flatten(),
            info_clean["gear"].flatten(),
            info_clean["is_sliding"].flatten(),
            info_clean["has_ground_contact"].flatten()
        ])

    def close(self):
        if self.init:
            self.conn1.send(99) # close signal
            self.init = False