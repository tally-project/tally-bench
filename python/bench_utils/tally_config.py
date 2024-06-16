from bench_utils.tally import TallyConfig

# default_tally_config = TallyConfig("priority", max_allowed_latency=0.1)
default_tally_config = TallyConfig("priority", max_allowed_latency=0.0316)

sensitivity_analysis_configs = [
    default_tally_config,
    TallyConfig("priority", disable_transformation=True), # for sensitivity study
    TallyConfig("priority", use_space_share=True),        # for sensitivity study
]