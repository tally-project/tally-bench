import pandas as pd

new_result_file = 'tally-bench-results/priority-aware-perf.csv'
old_result_file = 'tally-bench-results/priority-aware-perf-old.csv'

new_result = pd.read_csv(new_result_file)
old_result = pd.read_csv(old_result_file)

merged = pd.merge(new_result, old_result, on='exp_key', how='inner', suffixes=('_new', '_old'))

merged = merged[[
    "exp_key", "best_effort_tally_throughput_old" ,"best_effort_tally_throughput_new",
    "high_priority_orig_99th_latency_old",
    "high_priority_tally_99th_latency_old", "high_priority_tally_99th_latency_new",
    "preemption_latency_limit"
]]

merged["old_99th_latency_increase"] = merged["high_priority_tally_99th_latency_old"] / merged["high_priority_orig_99th_latency_old"]
merged["new_99th_latency_increase"] = merged["high_priority_tally_99th_latency_new"] / merged["high_priority_orig_99th_latency_old"]
merged["throughput_diff"] = merged["best_effort_tally_throughput_new"] - merged["best_effort_tally_throughput_old"]
# merged = merged.sort_values(by=['old_99th_latency_increase'], ascending=False)

merged = merged.round(3)

merged.to_csv('compare.csv', index=False) 