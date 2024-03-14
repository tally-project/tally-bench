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
                 min_wait_time=0., use_space_share=False, ptb_max_threads_per_sm=None,
                 fallback_to_original_threshold=None, min_worker_threshold=None,
                 latency_calculation_factor=None):
        
        self.scheduler_policy = scheduler_policy
        self.max_allowed_latency = max_allowed_latency
        self.use_original_configs = use_original_configs
        self.min_wait_time = min_wait_time
        self.use_space_share = use_space_share
        self.ptb_max_threads_per_sm = ptb_max_threads_per_sm
        self.fallback_to_original_threshold = fallback_to_original_threshold
        self.min_worker_threshold = min_worker_threshold
        self.latency_calculation_factor = latency_calculation_factor

    def to_dict(self):
        config = {
            "SCHEDULER_POLICY": self.scheduler_policy.upper(),
            "PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS": str(self.max_allowed_latency),
            "PRIORITY_USE_ORIGINAL_CONFIGS": str(self.use_original_configs).upper(),
            "PRIORITY_MIN_WAIT_TIME_MS": str(self.min_wait_time),
            "PRIORITY_USE_SPACE_SHARE": str(self.use_space_share).upper()
        }
        return config

def start_iox_roudi():
    logger.info("Starting Iox Roudi ...")
    start_iox_cmd = f"bash {iox_roudi_start_script} &"
    execute_cmd(start_iox_cmd)
    time.sleep(10)

def shut_down_iox_roudi():
    logger.info("Shutting down Iox Roudi ...")
    stop_iox_cmd = f"bash {iox_roudi_kill_script}"
    execute_cmd(stop_iox_cmd)

def start_tally(config: TallyConfig = None):
    logger.info("Starting Tally server ...")
    start_tally_cmd = f"bash {tally_start_script} &"

    if config:
        config_dict = config.to_dict()
        logger.info(f"Using Tally config: {config_dict}")
        for key in config_dict:
            start_tally_cmd = f"{key}={config_dict[key]} {start_tally_cmd}"
            
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
