import os
import sys
import logging
import script.tm_classes.tm_classes as tm

FULL_STEER = 65536

steer_mapping = {
    -2: -FULL_STEER,  # full left
    -1: -FULL_STEER // 2,  # left 1
     0: 0,   # straight
     1: FULL_STEER // 2,   # right 1
     2: FULL_STEER, # full right
}

def server_connection_process(connection, iface: tm.TMInterface, client, blind_mode) -> None:
    logger = logging.getLogger('__main__')
    infos = tm.TMInfos()
    success = iface.register(client)
    data = None
    if success:
        try:
            connected = True
            connection.send("ready")
            logger.info("Client connected to server.")
            if blind_mode:
                logger.info("Client mode : blind")
            else:
                logger.info("Client mode : road")
            while connected:
                data = connection.recv()
                if data is None:
                    continue
                # Data to actions
                elif data == 0: # no accel no steer
                    iface.set_input_state(accelerate=False, steer=steer_mapping[0], brake=False)
                elif data == 1: # no accel steer left 1
                    iface.set_input_state(accelerate=False, steer=steer_mapping[-1], brake=False)
                elif data == 2: # no accel steer left 2
                    iface.set_input_state(accelerate=False, steer=steer_mapping[-2], brake=False)
                elif data == 3: # no accel steer right 2
                    iface.set_input_state(accelerate=False, steer=steer_mapping[2], brake=False)
                elif data == 4: # no accel steer right 1
                    iface.set_input_state(accelerate=False, steer=steer_mapping[1], brake=False)
                elif data == 5: # accel no steer
                    iface.set_input_state(accelerate=True, steer=steer_mapping[0], brake=False)
                elif data == 6: # accel steer right 1
                    iface.set_input_state(accelerate=True, steer=steer_mapping[1], brake=False)
                elif data == 7: # accel steer right 2
                    iface.set_input_state(accelerate=True, steer=steer_mapping[2], brake=False)
                elif data == 8: # accel steer left 2
                    iface.set_input_state(accelerate=True, steer=steer_mapping[-2], brake=False)
                elif data == 9: # accel steer left 1
                    iface.set_input_state(accelerate=True, steer=steer_mapping[-1], brake=False)
                elif data == -1: # respawn
                    iface.give_up()
                elif data == -2: # get info
                    try:
                        if blind_mode:
                            info = infos.get_info_blind(iface.get_simulation_state())
                        else:
                            info = infos.get_info_road(iface.get_simulation_state())
                        connection.send(info)
                    except Exception as e:
                        logger.error(f"Error getting info: {e}")
                        connection.send(e)
                elif data == 99: # close signal
                    iface.close()
                    logger.debug("Shutting down client...")
                    connected = False
                data = None
            # time.sleep(0.0001) # 0.1 ms
        except Exception:
            logger.exception("Error in server_connection_process")
        logger.debug("Client stopped.")
    else:
        logger.error("Failed to register client")