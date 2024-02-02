import time

from utils.util import execute_cmd
from utils.util import logger

iox_roudi_start_script = "./tally/scripts/start_iox.sh"
iox_roudi_kill_script = "./tally/scripts/kill_iox.sh"

tally_start_script = "./tally/scripts/start_server.sh"
tally_kill_script = "./tally/scripts/kill_server.sh"

tally_query_script = "./tally/scripts/query_server.sh"

tally_client_script = "./tally/scripts/start_client.sh"
tally_client_local_script = "./tally/scripts/start_client_local.sh"

def start_iox_roudi():
    logger.info("Starting Iox Roudi ...")
    start_iox_cmd = f"bash {iox_roudi_start_script} &"
    execute_cmd(start_iox_cmd)
    time.sleep(5)

def shut_down_iox_roudi():
    logger.info("Shutting down Iox Roudi ...")
    stop_iox_cmd = f"bash {iox_roudi_kill_script}"
    execute_cmd(stop_iox_cmd)

def start_tally(preemption_limit=None):
    logger.info("Starting Tally server ...")
    start_tally_cmd = f"bash {tally_start_script} &"

    if preemption_limit:
        start_tally_cmd = f"PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS={preemption_limit} {start_tally_cmd}"

    execute_cmd(start_tally_cmd)
    time.sleep(3)

def shut_down_tally():
    logger.info("Shutting down Tally server ...")
    stop_tally_cmd = f"bash {tally_kill_script}"
    execute_cmd(stop_tally_cmd)

def query_tally():
    query_tally_cmd = f"bash {tally_query_script}"
    _, _, rc = execute_cmd(query_tally_cmd, get_output=True)
    return rc
