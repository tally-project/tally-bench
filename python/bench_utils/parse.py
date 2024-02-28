import pandas as pd
import numpy as np

from bench_utils.utils import load_json_from_file


def compute_relative_tp(run_res, baseline_res):
    run_throughput = run_res["iters"] / run_res["time_elapsed"]
    baseline_throughput = baseline_res["iters"] / baseline_res["time_elapsed"]
    return round(run_throughput / baseline_throughput, 2)


def compute_avg_latency(run_res):
    
    avg_latency = sum(run_res["latencies"]) / len(run_res["latencies"])
    return round(avg_latency, 3)


def compute_percentile_latency(run_res, percentile):
    
    latencies = run_res["latencies"]
    if len(latencies) > 10:
        latencies = latencies[min(10, len(latencies) // 2):]
    percentile_latency = np.percentile(latencies, percentile)
    return round(percentile_latency, 3)


def get_workload_type(key):
    if "train" in key:
        return "training"
    elif "infer" in key:
        for inference_type in ["single-stream", "server"]:
            if inference_type in key:
                return f"inference-{inference_type}"
    
    return "unknown"


def is_latency_critical(key):
    for inference_type in ["single-stream", "server"]:
        if inference_type in key:
            return True
    return False


def parse_result(file_name, single_job_result_out=None, priority_result_out=None):
    res = load_json_from_file(file_name)

    # All possible results
    default_res = res.get("default", {})
    mps_res = res.get("mps", {})
    tally_naive_res = res.get("tally_naive", {})
    tally_priority_res = res.get("tally_priority", {})

    # First Analysis: Compare single-job performance
    # This is more about getting a sense of the overhead of Tally and kernel transformation
    # CPU-bounded models generally have severely impacted performance when running on Tally

    single_job_default_perf = {}
    single_job_result = []
    single_job_keys = default_res.keys()
    
    for key in single_job_keys:

        measurment = default_res[key]["measurements"][0]
        metrics = measurment["metrics"]
        
        result_row = {
            "exp_key": key,
            "workload_type": "",
            "mem": metrics["gmem"],
            "mem_util": metrics["gmem_util"],
            "gpu_util": metrics["gpu_util"],
            "original_avg_latency": "",
            "original_throughput": "",
            "tally_naive_throughput": "",
        }

        key_numbered = f"{key}_0"

        original_run_res = default_res.get(key, {}).get("measurements", [{}])[0].get(key_numbered, {})
        tally_naive_run_res = tally_naive_res.get(key, {}).get("measurements", [{}])[0].get(key_numbered, {})

        single_job_default_perf[key] = original_run_res

        result_row["original_throughput"] = 1
        result_row["workload_type"] = get_workload_type(key)
        
        if any([type in result_row["workload_type"] for type in ["single-stream", "server"]]):
            result_row["original_avg_latency"] = compute_avg_latency(original_run_res)
            
        if "server" not in result_row["workload_type"]:
            if tally_naive_run_res:
                result_row["tally_naive_throughput"] = compute_relative_tp(tally_naive_run_res, original_run_res)

        single_job_result.append(result_row)

    if single_job_result_out:
        single_job_df = pd.DataFrame(single_job_result)
        single_job_df = single_job_df.sort_values(by=['workload_type', 'exp_key'], ascending=False)
        single_job_df.to_csv(single_job_result_out, index=True)

    # Second Anaylsis: Compare Tally vs MPS for co-located job performance
    latency_critical_result = []
    for key in tally_priority_res.keys():

        if key in single_job_keys:
            continue

        mps_run_res = mps_res.get(key, {})
        tally_priority_run_res = tally_priority_res.get(key, {})
    
        tally_measurments = tally_priority_run_res.get("measurements", [])
        mps_measurments = mps_run_res.get("measurements", [{}])
        if not tally_measurments:
            continue

        tally_measurement = tally_measurments[0]
        mps_measurment = mps_measurments[0]
        measurment_keys = list(tally_measurement.keys())
        
        if "error" in measurment_keys:
            continue

        measurment_keys.remove("metrics")

        job_1 = measurment_keys[0]
        job_2 = measurment_keys[1]

        job_1_clean = job_1.rsplit("_", 1)[0]
        job_2_clean = job_2.rsplit("_", 1)[0]

        assert(job_1_clean in default_res and job_2_clean in default_res)

        for tally_measurement in tally_measurments:

            if job_1 not in tally_measurement or job_2 not in tally_measurement:
                continue

            job_1_priority = tally_measurement[job_1]["priority"]
            job_2_priority = tally_measurement[job_2]["priority"]

            high_priority_job = job_1 if job_1_priority > job_2_priority else job_2
            best_effort_job = job_1 if job_1_priority < job_2_priority else job_2

            high_priority_job_clean = job_1_clean if high_priority_job == job_1 else job_2_clean
            best_effort_job_clean = job_1_clean if best_effort_job == job_1 else job_2_clean

            tally_config = tally_measurement["tally_config"]
            max_allowed_latency = tally_config.get("PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS")
            min_wait_time = tally_config.get("PRIORITY_MIN_WAIT_TIME_MS")
            use_original_configs = tally_config.get("PRIORITY_USE_ORIGINAL_CONFIGS")
            use_space_share = tally_config.get("PRIORITY_USE_SPACE_SHARE")

            lc_result_row = {
                "exp_key": key,
                "preemption_latency_limit": max_allowed_latency,
                "min_wait_time": min_wait_time,
                "use_original_configs" : use_original_configs,
                "use_space_share" : use_space_share,
                "high_priority_job": high_priority_job_clean,
                "high_priority_job_workload_type": get_workload_type(high_priority_job_clean),
                "best_effort_job": best_effort_job_clean,
                "best_effort_job_workload_type": get_workload_type(best_effort_job_clean),
                "best_effort_tally_throughput": "",
                "best_effort_mps_throughput": "",
                "high_priority_orig_avg_latency": "",
                "high_priority_tally_avg_latency": "",
                "high_priority_mps_avg_latency": "",
                "high_priority_orig_90th_latency": "",
                "high_priority_tally_90th_latency": "",
                "high_priority_mps_90th_latency": "",
                "high_priority_orig_95th_latency": "",
                "high_priority_tally_95th_latency": "",
                "high_priority_mps_95th_latency": "",
                "high_priority_orig_99th_latency": "",
                "high_priority_tally_99th_latency": "",
                "high_priority_mps_99th_latency": "",
            }

            lc_result_row["best_effort_tally_throughput"] = compute_relative_tp(tally_measurement[best_effort_job], single_job_default_perf[best_effort_job_clean])
            if mps_measurment:
                lc_result_row["best_effort_mps_throughput"] = compute_relative_tp(mps_measurment[best_effort_job], single_job_default_perf[best_effort_job_clean])

            lc_result_row["high_priority_orig_avg_latency"] = compute_avg_latency(single_job_default_perf[high_priority_job_clean])
            lc_result_row["high_priority_tally_avg_latency"] = compute_avg_latency(tally_measurement[high_priority_job])
            if mps_run_res:
                lc_result_row["high_priority_mps_avg_latency"] = compute_avg_latency(mps_measurment[high_priority_job])

            for percentile in [90, 95, 99]:
                lc_result_row[f"high_priority_orig_{percentile}th_latency"] = compute_percentile_latency(single_job_default_perf[high_priority_job_clean], percentile)
                lc_result_row[f"high_priority_tally_{percentile}th_latency"] = compute_percentile_latency(tally_measurement[high_priority_job], percentile)

                if mps_run_res:
                    lc_result_row[f"high_priority_mps_{percentile}th_latency"] = compute_percentile_latency(mps_measurment[high_priority_job], percentile)

            latency_critical_result.append(lc_result_row)

    if priority_result_out and latency_critical_result:
        lc_df = pd.DataFrame(latency_critical_result)
        lc_df = lc_df.sort_values(by=['high_priority_job', "exp_key"], ascending=False)
        lc_df.to_csv(priority_result_out, index=True)

    return single_job_result, latency_critical_result

if __name__ == "__main__":
    parse_result("result.json")