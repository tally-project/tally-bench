import pandas as pd
import numpy as np

from bench_utils.utils import load_json_from_file, compute_avg, compute_percentile


def compute_relative_tp(run_res, baseline_res):
    run_throughput = run_res["iters"] / run_res["time_elapsed"]
    baseline_throughput = baseline_res["iters"] / baseline_res["time_elapsed"]
    return round(run_throughput / baseline_throughput, 2)


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
    baseline_res = res.get("default", {})
    mps_res = res.get("mps", {})
    mps_priority_res = res.get("mps-priority", {})
    tally_naive_res = res.get("tally_naive", {})
    tally_priority_res = res.get("tally_priority", {})
    tgs_res = res.get("tgs", {})

    # First Analysis: Compare single-job performance
    # This is more about getting a sense of the overhead of Tally and kernel transformation
    # CPU-bounded models generally have severely impacted performance when running on Tally

    single_job_baseline_perf = {}
    single_job_result = []
    single_job_keys = [key for key in baseline_res.keys() if not ("infer" in key and "train" in key)]
    
    for key in single_job_keys:

        measurment = baseline_res[key]["measurements"][0]
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

        original_run_res = baseline_res.get(key, {}).get("measurements", [{}])[0].get(key_numbered, {})
        tally_naive_run_res = tally_naive_res.get(key, {}).get("measurements", [{}])[0].get(key_numbered, {})

        single_job_baseline_perf[key] = original_run_res

        result_row["original_throughput"] = 1
        result_row["workload_type"] = get_workload_type(key)
        
        if any([type in result_row["workload_type"] for type in ["single-stream", "server"]]):
            result_row["original_avg_latency"] = compute_avg(original_run_res["latencies"])
            
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

        if "amp" in key:
            continue

        mps_run_res = mps_res.get(key, {})
        mps_priority_run_res = mps_priority_res.get(key, {})
        time_slicing_run_res = baseline_res.get(key, {})
        tally_priority_run_res = tally_priority_res.get(key, {})
        tgs_run_res = tgs_res.get(key, {})
    
        tally_measurments = tally_priority_run_res.get("measurements", [])
        mps_measurments = mps_run_res.get("measurements", [{}])
        mps_priority_measurments = mps_priority_run_res.get("measurements", [{}])
        time_slicing_measurments = time_slicing_run_res.get("measurements", [{}])
        tgs_measurments = tgs_run_res.get("measurements", [{}])
        if not tally_measurments:
            continue

        tally_measurement = tally_measurments[0]
        mps_measurment = mps_measurments[0]
        mps_priority_measurment = mps_priority_measurments[0]
        time_slicing_measurment = time_slicing_measurments[0]
        tgs_measurment = tgs_measurments[0]
        measurment_keys = list(tally_measurement.keys())
        
        if "error" in measurment_keys:
            continue

        measurment_keys.remove("metrics")
        measurment_keys.remove("tally_config")

        job_1 = measurment_keys[0]
        job_2 = measurment_keys[1]

        job_1_clean = job_1.rsplit("_", 1)[0]
        job_2_clean = job_2.rsplit("_", 1)[0]

        
        if not (job_1_clean in baseline_res and job_2_clean in baseline_res):
            continue

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
            use_original_configs = tally_config.get("PRIORITY_USE_ORIGINAL_CONFIGS")
            use_space_share = tally_config.get("PRIORITY_USE_SPACE_SHARE")
            min_wait_time = tally_config.get("PRIORITY_MIN_WAIT_TIME_MS", "Default")
            disable_transformation = tally_config.get("PRIORITY_DISABLE_TRANSFORMATION", False)
            wait_time_to_use_original = tally_config.get("PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS", "Default")

            lc_result_row = {
                "exp_key": key,
                "preemption_latency_limit": max_allowed_latency,
                "min_wait_time": min_wait_time,
                "use_original_configs" : use_original_configs,
                "use_space_share" : use_space_share,
                "disable_transformation": disable_transformation,
                "wait_time_to_use_original": wait_time_to_use_original,
                "high_priority_job": high_priority_job_clean,
                "high_priority_job_workload_type": get_workload_type(high_priority_job_clean),
                "best_effort_job": best_effort_job_clean,
                "best_effort_job_workload_type": get_workload_type(best_effort_job_clean),
                "best_effort_tally_throughput": "",
                "best_effort_mps_throughput": "",
                "best_effort_mps_priority_throughput": "",
                "best_effort_time_slicing_throughput": "",
                "best_effort_tgs_throughput": "",
                "high_priority_tally_throughput": "",
                "high_priority_mps_throughput": "",
                "high_priority_mps_priority_throughput": "",
                "high_priority_time_slicing_throughput": "",
                "high_priority_tgs_throughput": "",
                "high_priority_orig_avg_latency": "",
                "high_priority_tally_avg_latency": "",
                "high_priority_mps_avg_latency": "",
                "high_priority_mps_priority_avg_latency": "",
                "high_priority_time_slicing_avg_latency": "",
                "high_priority_tgs_avg_latency": "",
                "high_priority_orig_90th_latency": "",
                "high_priority_tally_90th_latency": "",
                "high_priority_mps_90th_latency": "",
                "high_priority_mps_priority_90th_latency": "",
                "high_priority_time_slicing_90th_latency": "",
                "high_priority_tgs_90th_latency": "",
                "high_priority_orig_95th_latency": "",
                "high_priority_tally_95th_latency": "",
                "high_priority_mps_95th_latency": "",
                "high_priority_mps_priority_95th_latency": "",
                "high_priority_time_slicing_95th_latency": "",
                "high_priority_tgs_priority_95th_latency": "",
                "high_priority_orig_99th_latency": "",
                "high_priority_tally_99th_latency": "",
                "high_priority_mps_99th_latency": "",
                "high_priority_mps_priority_99th_latency": "",
                "high_priority_time_slicing_99th_latency": "",
                "high_priority_tgs_99th_latency": "",
            }

            single_job_baseline_perf_key = high_priority_job_clean
            if "server" in single_job_baseline_perf_key:
                single_job_baseline_perf_key = single_job_baseline_perf_key.split("infer_")[0] + "infer_single-stream_1"
                
            high_priority_baseline_perf = single_job_baseline_perf[high_priority_job_clean]
            high_priority_baseline_latency_perf = single_job_baseline_perf[single_job_baseline_perf_key]
            best_effort_baseline_perf = single_job_baseline_perf[best_effort_job_clean]

            lc_result_row["best_effort_tally_throughput"] = compute_relative_tp(tally_measurement[best_effort_job], best_effort_baseline_perf)
            if mps_measurment:
                lc_result_row["best_effort_mps_throughput"] = compute_relative_tp(mps_measurment[best_effort_job], best_effort_baseline_perf)
            if mps_priority_measurment:
                lc_result_row["best_effort_mps_priority_throughput"] = compute_relative_tp(mps_priority_measurment[best_effort_job], best_effort_baseline_perf)
            if time_slicing_measurment:
                lc_result_row["best_effort_time_slicing_throughput"] = compute_relative_tp(time_slicing_measurment[best_effort_job], best_effort_baseline_perf)
            if tgs_measurment:
                lc_result_row["best_effort_tgs_throughput"] = compute_relative_tp(tgs_measurment[best_effort_job], best_effort_baseline_perf)

            lc_result_row["high_priority_tally_throughput"] = compute_relative_tp(tally_measurement[high_priority_job], high_priority_baseline_perf)
            if mps_measurment:
                lc_result_row["high_priority_mps_throughput"] = compute_relative_tp(mps_measurment[high_priority_job], high_priority_baseline_perf)
            if mps_priority_measurment:
                lc_result_row["high_priority_mps_priority_throughput"] = compute_relative_tp(mps_priority_measurment[high_priority_job], high_priority_baseline_perf)
            if time_slicing_measurment:
                lc_result_row["high_priority_time_slicing_throughput"] = compute_relative_tp(time_slicing_measurment[high_priority_job], high_priority_baseline_perf)
            if tgs_measurment:
                lc_result_row["high_priority_tgs_throughput"] = compute_relative_tp(tgs_measurment[high_priority_job], high_priority_baseline_perf)

            lc_result_row["high_priority_orig_avg_latency"] = compute_avg(high_priority_baseline_latency_perf["latencies"])
            lc_result_row["high_priority_tally_avg_latency"] = compute_avg(tally_measurement[high_priority_job]["latencies"])
            if mps_measurment:
                lc_result_row["high_priority_mps_avg_latency"] = compute_avg(mps_measurment[high_priority_job]["latencies"])
            if mps_priority_measurment:
                lc_result_row["high_priority_mps_priority_avg_latency"] = compute_avg(mps_priority_measurment[high_priority_job]["latencies"])
            if time_slicing_measurment:
                lc_result_row["high_priority_time_slicing_avg_latency"] = compute_avg(time_slicing_measurment[high_priority_job]["latencies"])
            if tgs_measurment:
                lc_result_row["high_priority_tgs_avg_latency"] = compute_avg(tgs_measurment[high_priority_job]["latencies"])

            for percentile in [90, 95, 99]:
                lc_result_row[f"high_priority_orig_{percentile}th_latency"] = compute_percentile(high_priority_baseline_latency_perf["latencies"], percentile)
                lc_result_row[f"high_priority_tally_{percentile}th_latency"] = compute_percentile(tally_measurement[high_priority_job]["latencies"], percentile)

                if mps_measurment:
                    lc_result_row[f"high_priority_mps_{percentile}th_latency"] = compute_percentile(mps_measurment[high_priority_job]["latencies"], percentile)
                if mps_priority_measurment:
                    lc_result_row[f"high_priority_mps_priority_{percentile}th_latency"] = compute_percentile(mps_priority_measurment[high_priority_job]["latencies"], percentile)
                if time_slicing_measurment:
                    lc_result_row[f"high_priority_time_slicing_{percentile}th_latency"] = compute_percentile(time_slicing_measurment[high_priority_job]["latencies"], percentile)
                if tgs_measurment:
                    lc_result_row[f"high_priority_tgs_{percentile}th_latency"] = compute_percentile(tgs_measurment[high_priority_job]["latencies"], percentile)

            latency_critical_result.append(lc_result_row)

    if priority_result_out and latency_critical_result:
        lc_df = pd.DataFrame(latency_critical_result)
        lc_df = lc_df.sort_values(by=['high_priority_job', "exp_key"], ascending=False)
        lc_df.to_csv(priority_result_out, index=True)

    return single_job_result, latency_critical_result


def get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_config, metric="avg"):

    high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
    baseline_latencies, time_sliced_latencies, mps_latencies, mps_priority_latencies, tgs_latencies = [], [], [], [], []
    tally_latencies, tally_space_share_latencies, tally_no_transform_latencies = [], [], []
    priority_time_sliced_throughputs, priority_mps_throughputs, priority_mps_priority_throughputs, priority_tgs_throughputs = [], [], [], []
    priority_tally_throughputs, priority_tally_space_share_throughputs, priority_tally_no_transform_throughputs = [], [], []
    time_sliced_throughputs, mps_throughputs, mps_priority_throughputs, tgs_throughputs = [], [], [], []
    tally_throughputs, tally_space_share_throughputs, tally_no_transform_throughputs = [], [], []
    used_best_effort_jobs = []

    for best_effort_job in best_effort_jobs:
        best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]

        if best_effort_job_df.empty:
            continue

        baseline_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
        time_sliced_latency = best_effort_job_df[f"high_priority_time_slicing_{metric}_latency"].values[0]
        priority_time_sliced_throughput = best_effort_job_df[f"high_priority_time_slicing_throughput"].values[0]
        time_sliced_throughput = best_effort_job_df[f"best_effort_time_slicing_throughput"].values[0]
        mps_latency = best_effort_job_df[f"high_priority_mps_{metric}_latency"].values[0]
        priority_mps_throughput = best_effort_job_df[f"high_priority_mps_throughput"].values[0]
        mps_throughput = best_effort_job_df[f"best_effort_mps_throughput"].values[0]
        mps_priority_latency = best_effort_job_df[f"high_priority_mps_priority_{metric}_latency"].values[0]
        priority_mps_priority_throughput = best_effort_job_df[f"high_priority_mps_priority_throughput"].values[0]
        mps_priority_throughput = best_effort_job_df[f"best_effort_mps_priority_throughput"].values[0]
        tgs_latency = best_effort_job_df[f"high_priority_tgs_{metric}_latency"].values[0]
        priority_tgs_throughput = best_effort_job_df[f"high_priority_tgs_throughput"].values[0]
        tgs_throughput = best_effort_job_df[f"best_effort_tgs_throughput"].values[0]

        tally_space_share_df = best_effort_job_df[best_effort_job_df[f"use_space_share"] == True]
        if not tally_space_share_df.empty:
            tally_space_share_latency = tally_space_share_df[f"high_priority_tally_{metric}_latency"].values[0]
            priority_tally_space_share_throughput = tally_space_share_df[f"best_effort_tally_throughput"].values[0]
            tally_space_share_throughput = tally_space_share_df[f"high_priority_tally_throughput"].values[0]
        else:
            tally_space_share_latency, priority_tally_space_share_throughput, tally_space_share_throughput = 0, 0, 0

        tally_no_transform_df = best_effort_job_df[best_effort_job_df[f"disable_transformation"] == True]
        if not tally_no_transform_df.empty:
            tally_no_transform_latency = tally_no_transform_df[f"high_priority_tally_{metric}_latency"].values[0]
            priority_tally_no_transform_throughput = tally_no_transform_df[f"best_effort_tally_throughput"].values[0]
            tally_no_transform_throughput = tally_no_transform_df[f"high_priority_tally_throughput"].values[0]
        else:
            tally_no_transform_latency, priority_tally_no_transform_throughput, tally_no_transform_throughput = 0, 0, 0

        tally_df = best_effort_job_df
        for param in tally_config:
            val = tally_config[param]
            tally_df = tally_df[tally_df[param] == val]

        if tally_df.empty:
            tally_latency, tally_throughput, priority_tally_throughput = 0., 0., 0.
        else:
            tally_latency = tally_df[f"high_priority_tally_{metric}_latency"].values[0]
            tally_throughput = tally_df[f"best_effort_tally_throughput"].values[0]
            priority_tally_throughput = tally_df[f"high_priority_tally_throughput"].values[0]

        baseline_latencies.append(baseline_latency)
        time_sliced_latencies.append(time_sliced_latency)
        mps_latencies.append(mps_latency)
        mps_priority_latencies.append(mps_priority_latency)
        tgs_latencies.append(tgs_latency)
        tally_latencies.append(tally_latency)
        tally_space_share_latencies.append(tally_space_share_latency)
        tally_no_transform_latencies.append(tally_no_transform_latency)

        priority_time_sliced_throughputs.append(priority_time_sliced_throughput)
        priority_mps_throughputs.append(priority_mps_throughput)
        priority_mps_priority_throughputs.append(priority_mps_priority_throughput)
        priority_tgs_throughputs.append(priority_tgs_throughput)
        priority_tally_throughputs.append(priority_tally_throughput)
        priority_tally_space_share_throughputs.append(priority_tally_space_share_throughput)
        priority_tally_no_transform_throughputs.append(priority_tally_no_transform_throughput)

        time_sliced_throughputs.append(time_sliced_throughput)
        mps_throughputs.append(mps_throughput)
        mps_priority_throughputs.append(mps_priority_throughput)
        tgs_throughputs.append(tgs_throughput)
        tally_throughputs.append(tally_throughput)
        tally_space_share_throughputs.append(tally_space_share_throughput)
        tally_no_transform_throughputs.append(tally_no_transform_throughput)

        used_best_effort_jobs.append(best_effort_job)
    
    data = {
        "baseline_latencies": baseline_latencies,
        "time_sliced_latencies": time_sliced_latencies,
        "mps_latencies": mps_latencies,
        "mps_priority_latencies": mps_priority_latencies,
        "tgs_latencies": tgs_latencies,
        "tally_latencies": tally_latencies,
        "tally_space_share_latencies": tally_space_share_latencies,
        "tally_no_transform_latencies": tally_no_transform_latencies,
        "priority_time_sliced_throughputs": priority_time_sliced_throughputs,
        "priority_mps_throughputs": priority_mps_throughputs,
        "priority_mps_priority_throughputs": priority_mps_priority_throughputs,
        "priority_tgs_throughputs": priority_tgs_throughputs,
        "priority_tally_throughputs": priority_tally_throughputs,
        "priority_tally_space_share_throughputs": priority_tally_space_share_throughputs,
        "priority_tally_no_transform_throughputs": priority_tally_no_transform_throughputs,
        "time_sliced_throughputs": time_sliced_throughputs,
        "mps_throughputs": mps_throughputs,
        "mps_priority_throughputs": mps_priority_throughputs,
        "tgs_throughputs": tgs_throughputs,
        "tally_throughputs": tally_throughputs,
        "tally_space_share_throughputs": tally_space_share_throughputs,
        "tally_no_transform_throughputs": tally_no_transform_throughputs,
        "used_best_effort_jobs": used_best_effort_jobs
    }

    return data