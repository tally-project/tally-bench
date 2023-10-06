import time

from utils.util import execute_cmd

iox_roudi_start_script = "./tally/scripts/start_iox.sh"
iox_roudi_kill_script = "./tally/scripts/kill_iox.sh"

tally_start_script = "./tally/scripts/start_server.sh"
tally_kill_script = "./tally/scripts/kill_server.sh"

tally_client_script = "./tally/scripts/start_client.sh"

def start_iox_roudi():
    print("Starting Iox Roudi ...")
    start_iox_cmd = f"bash {iox_roudi_start_script} &"
    execute_cmd(start_iox_cmd)
    time.sleep(5)

def shut_down_iox_roudi():
    print("Shutting down Iox Roudi ...")
    stop_iox_cmd = f"bash {iox_roudi_kill_script}"
    execute_cmd(stop_iox_cmd)

def start_tally():
    print("Starting Tally server ...")
    start_tally_cmd = f"bash {tally_start_script} &"
    execute_cmd(start_tally_cmd)

def shut_down_tally():
    print("Shutting down Tally server ...")
    stop_tally_cmd = f"bash {tally_kill_script}"
    execute_cmd(stop_tally_cmd)