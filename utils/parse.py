import pandas as pd
import numpy as np

from utils.util import load_json_from_file


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

def parse_result(file_name, single_job_result_out=None, throughput_result_out=None, priority_result_out=None):
    res = load_json_from_file(file_name)

    # All possible results
    default_res = res.get("default", {})
    mps_res = res.get("mps", {})
    tally_naive_res = res.get("tally_naive", {})
    tally_sharing_res = res.get("tally_workload_agnostic_sharing", {})
    tally_priority_res = res.get("tally_priority", {})

    # First Analysis: Compare single-job performance
    # This is more about getting a sense of the overhead of Tally and kernel transformation
    # CPU-bounded models generally have severely impacted performance when running on Tally

    single_job_default_perf = {}
    single_job_keys = []
    
    for key in list(default_res.keys()):
        if "offline" in key:
            continue

        job_keys = list(default_res[key].keys())
        job_keys.remove("metrics")

        if len(job_keys) == 1:
            single_job_keys.append(key)

    single_job_result = []
    for key in single_job_keys:

        metrics = default_res[key]["metrics"]
        
        result_row = {
            "exp_key": key,
            "workload_type": "",
            "mem": metrics["gmem"],
            "mem_util": metrics["gmem_util"],
            "gpu_util": metrics["gpu_util"],
            "original_throughput": "",
            "tally_naive_throughput": "",
            "tally_sharing_throughput": "",
            "tally_priority_throughput": "",
            "original_avg_latency": "",
            "tally_priority_avg_latency": ""
        }

        key_numbered = f"{key}_0"

        original_run_res = default_res.get(key, {}).get(key_numbered, {})
        tally_naive_run_res = tally_naive_res.get(key, {}).get(key_numbered, {})
        tally_sharing_run_res = tally_sharing_res.get(key, {}).get(key_numbered, {})
        tally_priority_run_res = tally_priority_res.get(key, {}).get(key_numbered, {})

        single_job_default_perf[key] = original_run_res

        result_row["original_throughput"] = 1
        result_row["workload_type"] = get_workload_type(key)
        
        if any([type in result_row["workload_type"] for type in ["single-stream", "server"]]):
            result_row["original_avg_latency"] = compute_avg_latency(original_run_res)
            if tally_priority_run_res:
                result_row["tally_priority_avg_latency"] = compute_avg_latency(tally_priority_run_res)
        
        if "server" not in result_row["workload_type"]:
            if tally_naive_run_res:
                result_row["tally_naive_throughput"] = compute_relative_tp(tally_naive_run_res, original_run_res)
            if tally_sharing_run_res:
                result_row["tally_sharing_throughput"] = compute_relative_tp(tally_sharing_run_res, original_run_res)

        single_job_result.append(result_row)

    if single_job_result_out:
        single_job_df = pd.DataFrame(single_job_result)
        single_job_df = single_job_df.sort_values(by=['workload_type', 'exp_key'], ascending=False)
        single_job_df.to_csv(single_job_result_out, index=True)
    

    # Second Anaylsis: Compare Tally vs MPS for co-located job performance
    co_locate_result = []
    latency_critical_result = []

    co_locate_keys = []
    for res in [mps_res, tally_naive_res, tally_sharing_res, tally_priority_res]:

        for key in res.keys():

            job_keys_list = None

            if res == tally_priority_res:

                if "metrics" in res[key] or "error" in res[key] or "profiled" in res[key]:
                    continue

                job_keys_list = [list(res[key][limit].keys()) for limit in res[key].keys()]
            else:
                job_keys_list = [list(res[key].keys())]

            for job_keys in job_keys_list:

                if "error" in job_keys or "profiled" in job_keys:
                    continue

                job_keys.remove("metrics")
                if len(job_keys) == 2 and key not in co_locate_keys:
                    co_locate_keys.append(key)

    for key in co_locate_keys:

        mps_run_res = mps_res.get(key, {})
        tally_naive_run_res = tally_naive_res.get(key, {})
        tally_sharing_run_res = tally_sharing_res.get(key, {})
        tally_priority_run_res = tally_priority_res.get(key, {})
        
        exits_run_res = None

        for run_res in [mps_run_res,
                        tally_naive_run_res,
                        tally_sharing_run_res,
                        tally_priority_run_res]:
            if run_res:
                exits_run_res = run_res

        assert(exits_run_res)

        if exits_run_res == tally_priority_run_res:
            exits_run_res = tally_priority_run_res[list(tally_priority_run_res.keys())[0]]

        exits_run_res_keys = list(exits_run_res.keys())
        if "error" in exits_run_res_keys:
            continue

        exits_run_res_keys.remove("metrics")

        job_1 = list(exits_run_res_keys)[0]
        job_2 = list(exits_run_res_keys)[1]

        job_1_clean = job_1.rsplit("_", 1)[0]
        job_2_clean = job_2.rsplit("_", 1)[0]

        assert(job_1_clean in default_res and job_2_clean in default_res)

        job_1_default_res = single_job_default_perf[job_1_clean]
        job_2_default_res = single_job_default_perf[job_2_clean]

        job_1_workload_type = get_workload_type(job_1_clean)
        job_2_workload_type = get_workload_type(job_2_clean)

        if not is_latency_critical(job_1_clean) and not is_latency_critical(job_2_clean):

            co_locate_result_row = {
                "exp_key": key,
                "job_1": job_1_clean,
                "job_1_workload_type": job_1_workload_type,
                "job_2": job_2_clean,
                "job_2_workload_type": job_2_workload_type,
                "job_1_mps_throughput": "", "job_2_mps_throughput": "",
                "job_1_tally_throughput": "", "job_2_tally_throughput": ""
            }

            if mps_run_res:
                co_locate_result_row["job_1_mps_throughput"] = compute_relative_tp(mps_run_res[job_1], job_1_default_res)
                co_locate_result_row["job_2_mps_throughput"] = compute_relative_tp(mps_run_res[job_2], job_2_default_res)
            if tally_sharing_run_res:
                co_locate_result_row["job_1_tally_throughput"] = compute_relative_tp(tally_sharing_run_res[job_1], job_1_default_res)
                co_locate_result_row["job_2_tally_throughput"] = compute_relative_tp(tally_sharing_run_res[job_2], job_2_default_res)
        
            co_locate_result.append(co_locate_result_row)

        if tally_priority_run_res:

            preemption_latency_limits = list(tally_priority_run_res.keys())
    
            for limit in preemption_latency_limits:

                tally_priority_limit_run_res = tally_priority_run_res[limit]

                if "error" in tally_priority_limit_run_res:
                    continue

                job_1_priority = tally_priority_limit_run_res[job_1]["priority"]
                job_2_priority = tally_priority_limit_run_res[job_2]["priority"]

                high_priority_job = job_1 if job_1_priority > job_2_priority else job_2
                best_effort_job = job_1 if job_1_priority < job_2_priority else job_2

                high_priority_job_clean = job_1_clean if high_priority_job == job_1 else job_2_clean
                best_effort_job_clean = job_1_clean if best_effort_job == job_1 else job_2_clean

                lc_result_row = {
                    "exp_key": key,
                    "preemption_latency_limit": limit,
                    "high_priority_job": high_priority_job_clean,
                    "high_priority_job_workload_type": get_workload_type(high_priority_job_clean),
                    "best_effort_job": best_effort_job_clean,
                    "best_effort_job_workload_type": get_workload_type(best_effort_job_clean),
                    "best_effort_tally_throughput": compute_relative_tp(tally_priority_limit_run_res[best_effort_job], single_job_default_perf[best_effort_job_clean]),
                }

                lc_result_row["high_priority_orig_avg_latency"] = compute_avg_latency(single_job_default_perf[high_priority_job_clean])
                lc_result_row["high_priority_tally_avg_latency"] = compute_avg_latency(tally_priority_limit_run_res[high_priority_job])

                for percentile in [90, 95, 99]:
                    lc_result_row[f"high_priority_orig_{percentile}th_latency"] = compute_percentile_latency(single_job_default_perf[high_priority_job_clean], percentile)
                    lc_result_row[f"high_priority_tally_{percentile}th_latency"] = compute_percentile_latency(tally_priority_limit_run_res[high_priority_job], percentile)

                latency_critical_result.append(lc_result_row)

    if throughput_result_out and co_locate_result:
        co_locate_df = pd.DataFrame(co_locate_result)
        co_locate_df["mps_throughput_sum"] = pd.to_numeric((co_locate_df["job_1_mps_throughput"] + co_locate_df["job_2_mps_throughput"])).round(2)
        co_locate_df["tally_throughput_sum"] = pd.to_numeric((co_locate_df["job_1_tally_throughput"] + co_locate_df["job_2_tally_throughput"])).round(2)
        co_locate_df["tally_speedup"] = pd.to_numeric((co_locate_df["tally_throughput_sum"] / co_locate_df["mps_throughput_sum"])).round(2)
        co_locate_df = co_locate_df.sort_values(by='tally_speedup', ascending=False)
        co_locate_df.to_csv(throughput_result_out, index=True)

    if priority_result_out and latency_critical_result:
        lc_df = pd.DataFrame(latency_critical_result)
        lc_df = lc_df.sort_values(by=['high_priority_job', "exp_key", "preemption_latency_limit"], ascending=False)
        lc_df.to_csv(priority_result_out, index=True)

    return single_job_result, co_locate_result

if __name__ == "__main__":
    parse_result("result.json")