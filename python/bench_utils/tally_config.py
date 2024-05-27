from bench_utils.tally import TallyConfig

sensitivity_analysis_configs = [
    TallyConfig("priority", max_allowed_latency=0.00316), # 10^-3 * sqrt(10)
    TallyConfig("priority", max_allowed_latency=0.01),    # 10^-2
    TallyConfig("priority", max_allowed_latency=0.0316),  # 10^-2 * sqrt(10)
    TallyConfig("priority", max_allowed_latency=0.1),     # 10^-1
    TallyConfig("priority", max_allowed_latency=0.316),   # 10^-1 * sqrt(10)
    TallyConfig("priority", max_allowed_latency=1.0),     # 10^0
    TallyConfig("priority", disable_transformation=True), # for sensitivity study
    TallyConfig("priority", use_space_share=True),        # for sensitivity study
]

default_tally_config = TallyConfig("priority", max_allowed_latency=0.1)