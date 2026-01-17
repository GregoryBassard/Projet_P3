from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import math


class TMLogger(Client):
    def __init__(self):
        super().__init__()

        self.road_pos = []
        self.saved_flag = True

    def on_registered(self, iface: TMInterface):
        print("Connecté à TMInterface")

    def on_run_step(self, iface: TMInterface, _time: int):
        # Récupérer l’état du joueur
        state = iface.get_simulation_state()

        pos = state.position

        # Position absolue
        pos_x, pos_y, pos_z = pos[0], pos[1], pos[2]

        # Exemple d’affichage
        print(
            f"Pos=({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})"
        )

        # # check if pos (+/-2m) is not in self.road_pos
        # append_point = True
        # if len(self.road_pos) == 0:
        #     pos_to_add = [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]
        #     self.road_pos.append(pos_to_add)
        #     print(f"Added road point: {pos_to_add}")
        # else:
        #     for road_point in self.road_pos:
        #         distance = math.sqrt(
        #             (pos_x - road_point[0]) ** 2 +
        #             (pos_y - road_point[1]) ** 2 +
        #             (pos_z - road_point[2]) ** 2
        #         )
        #         if distance < 2.0:
        #             append_point = False
        #     if append_point:
        #         pos_to_add = [round(pos_x, 3), round(pos_y, 3), round(pos_z, 3)]
        #         self.road_pos.append(pos_to_add)
        #         print(f"Added road point: {pos_to_add}")

        # if state.player_info.race_finished and self.saved_flag:
        #     # save road_pos to file road_pos.txt
        #     with open("road_pos.txt", "w") as f:
        #         for point in self.road_pos:
        #             f.write(f"{point[0]}, {point[1]}, {point[2]}\n")
        #     print("Road points saved to road_pos.txt")
        #     self.saved_flag = False

    def on_shutdown(self, iface: TMInterface):
        print("Déconnexion")


if __name__ == "__main__":
    run_client(TMLogger())
