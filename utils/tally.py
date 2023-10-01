import time

from utils.util import execute_cmd

iox_roudi_start_script = "./tally/scripts/start_iox_roudi.sh"
iox_roudi_kill_script = "./tally/scripts/kill_iox_roudi.sh"

tally_start_script = "./tally/scripts/start_tally_server.sh"
tally_kill_script = "./tally/scripts/kill_tally_server.sh"

tally_client_script = "./tally/scripts/start_tally_client.sh"

def start_tally():
    print("Starting Tally server ...")
    time.sleep(3)

    start_iox_cmd = f"bash {iox_roudi_start_script} &"
    start_tally_cmd = f"bash {tally_start_script} &"

    for cmd in [start_iox_cmd, start_tally_cmd]:
        execute_cmd(cmd)
        time.sleep(3)

def shut_down_tally():
    print("Shutting down Tally server ...")
    time.sleep(3)

    stop_iox_cmd = f"bash {iox_roudi_kill_script}"
    stop_tally_cmd = f"bash {tally_kill_script}"

    for cmd in [stop_tally_cmd, stop_iox_cmd]:
        execute_cmd(cmd)
        time.sleep(1)