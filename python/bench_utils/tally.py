import time

from bench_utils.utils import execute_cmd
from bench_utils.utils import logger

iox_roudi_start_script = "./tally/scripts/start_iox.sh"
iox_roudi_kill_script = "./tally/scripts/kill_iox.sh"

tally_start_script = "./tally/scripts/start_server.sh"
tally_kill_script = "./tally/scripts/kill_server.sh"

tally_query_script = "./tally/scripts/query_server.sh"

tally_client_script = "./tally/scripts/start_client.sh"
tally_client_local_script = "./tally/scripts/start_client_local.sh"

class TallyConfig:

    def __init__(self, scheduler_policy, max_allowed_latency=0.1, use_original_configs=False,
                use_space_share=False, min_wait_time=None, wait_time_to_use_original=None,
                disable_transformation=None):
        
        self.scheduler_policy = scheduler_policy
        self.max_allowed_latency = max_allowed_latency
        self.use_original_configs = use_original_configs
        self.min_wait_time = min_wait_time
        self.use_space_share = use_space_share
        self.wait_time_to_use_original = wait_time_to_use_original
        self.disable_transformation = disable_transformation

    def to_dict(self):
        config = {
            "SCHEDULER_POLICY": self.scheduler_policy.upper(),
            "PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS": str(self.max_allowed_latency),
            "PRIORITY_USE_ORIGINAL_CONFIGS": str(self.use_original_configs).upper(),
            "PRIORITY_USE_SPACE_SHARE": str(self.use_space_share).upper()
        }

        if self.disable_transformation is not None:
            config["PRIORITY_DISABLE_TRANSFORMATION"] = str(self.disable_transformation).upper()

        if self.min_wait_time is not None:
            config["PRIORITY_MIN_WAIT_TIME_MS"] = str(self.min_wait_time)

        if self.wait_time_to_use_original is not None:
            config["PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS"] = str(self.wait_time_to_use_original)

        return config

def start_iox_roudi():
    logger.info("Starting Iox Roudi ...")
    start_iox_cmd = f"bash {iox_roudi_start_script} &"
    execute_cmd(start_iox_cmd)
    time.sleep(15)

def shut_down_iox_roudi():
    logger.info("Shutting down Iox Roudi ...")
    stop_iox_cmd = f"bash {iox_roudi_kill_script}"
    execute_cmd(stop_iox_cmd)

def start_tally(config: TallyConfig = None, use_tgs=False):
    logger.info("Starting Tally server ...")
    start_tally_cmd = f"bash {tally_start_script} &"

    if config:
        config_dict = config.to_dict()
        logger.info(f"Using Tally config: {config_dict}")
        for key in config_dict:
            start_tally_cmd = f"{key}={config_dict[key]} {start_tally_cmd}"
            
    if use_tgs:
        start_tally_cmd = f"SCHEDULER_POLICY=TGS {start_tally_cmd}"

    execute_cmd(start_tally_cmd)
    time.sleep(2)

def shut_down_tally():
    logger.info("Shutting down Tally server ...")
    stop_tally_cmd = f"bash {tally_kill_script}"
    execute_cmd(stop_tally_cmd)

def query_tally():
    query_tally_cmd = f"bash {tally_query_script}"
    _, _, rc = execute_cmd(query_tally_cmd, get_output=True)
    return rc
