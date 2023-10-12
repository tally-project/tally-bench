import pandas as pd

from utils.util import load_json_from_file

def parse_result(file_name, single_job_result_out=None, co_locate_result_out=None):
    res = load_json_from_file(file_name)

    # All possible results
    default_res = res.get("default", {})
    mps_res = res.get("mps", {})
    tally_naive_res = res.get("tally_naive", {})
    tally_aware_res = res.get("tally_workload_aware_sharing", {})
    tally_agnostic_res = res.get("tally_workload_agnostic_sharing", {})

    compute_relative_throughput = lambda run_res, baseline : \
                                    round((run_res["iters"] / run_res["time_elapsed"]) / baseline, 2)

    # First Analysis: Compare single-job performance
    # This is more about getting a sense of the overhead of Tally and kernel transformation
    # CPU-bounded models generally have severely impacted performance when running on Tally

    single_job_perf = {}
    single_job_keys = list(default_res.keys())

    single_job_result = []
    for key in single_job_keys:
        
        result_row = {
            "model": key, "original": "", "tally_naive": "",
            "tally_workload_aware": "", "tally_workload_agnostic": ""
        }

        original_run_res = default_res.get(key, {}).get(key + "_0", {})
        tally_naive_run_res = tally_naive_res.get(key, {}).get(key + "_0", {})
        tally_aware_run_res = tally_aware_res.get(key, {}).get(key + "_0", {})
        tally_agnostic_run_res = tally_agnostic_res.get(key, {}).get(key + "_0", {})

        original_throughput = original_run_res["iters"] / original_run_res["time_elapsed"]
        single_job_perf[key] = original_throughput

        result_row["original"] = 1
        if tally_naive_run_res:
            result_row["tally_naive"] = compute_relative_throughput(tally_naive_run_res, original_throughput)
        if tally_aware_run_res:
            result_row["tally_workload_aware"] = compute_relative_throughput(tally_aware_run_res, original_throughput)
        if tally_agnostic_run_res:
            result_row["tally_workload_agnostic"] = compute_relative_throughput(tally_agnostic_run_res, original_throughput)
        
        single_job_result.append(result_row)
    
    if single_job_result_out:
        single_job_df = pd.DataFrame(single_job_result)
        single_job_df.to_csv(single_job_result_out, index=True)
    

    # Second Anaylsis: Compare Tally vs MPS for co-located job performance
    co_locate_result = []

    co_locate_keys = []
    for res in [mps_res, tally_naive_res, tally_aware_res, tally_agnostic_res]:
        for key in res.keys():
            if len(res[key]) == 2 and key not in co_locate_keys:
                co_locate_keys.append(key)

    for key in co_locate_keys:
        mps_run_res = mps_res.get(key, {})
        tally_naive_run_res = tally_naive_res.get(key, {})
        tally_aware_run_res = tally_aware_res.get(key, {})
        tally_agnostic_run_res = tally_agnostic_res.get(key, {})
        
        for run_res in [mps_run_res, tally_naive_run_res, tally_aware_run_res, tally_agnostic_run_res]:
            if run_res:
                exits_run_res = run_res

        assert(exits_run_res)

        job_1 = list(exits_run_res.keys())[0]
        job_2 = list(exits_run_res.keys())[1]

        job_1_clean = job_1.rsplit("_", 1)[0]
        job_2_clean = job_2.rsplit("_", 1)[0]

        assert(job_1_clean in default_res and job_2_clean in default_res)

        result_row = {
            "model_1": job_1_clean, "model_2": job_2_clean,
            "model_1_mps": "", "model_2_mps": "",
            "model_1_tally_naive": "", "model_2_tally_naive": "",
            "model_1_tally_aware": "", "model_2_tally_aware": "",
            "model_1_tally_agnostic": "", "model_2_tally_agnostic": "",
        }

        default_job_1_throughput = single_job_perf[job_1_clean]
        default_job_2_throughput = single_job_perf[job_2_clean]

        if mps_run_res:
            result_row["model_1_mps"] = compute_relative_throughput(mps_run_res[job_1], default_job_1_throughput)
            result_row["model_2_mps"] = compute_relative_throughput(mps_run_res[job_2], default_job_2_throughput)
        if tally_naive_run_res:
            result_row["model_1_tally_naive"] = compute_relative_throughput(tally_naive_run_res[job_1], default_job_1_throughput)
            result_row["model_2_tally_naive"] = compute_relative_throughput(tally_naive_run_res[job_2], default_job_2_throughput)
        if tally_aware_run_res:
            result_row["model_1_tally_aware"] = compute_relative_throughput(tally_aware_run_res[job_1], default_job_1_throughput)
            result_row["model_2_tally_aware"] = compute_relative_throughput(tally_aware_run_res[job_2], default_job_2_throughput)
        if tally_agnostic_run_res:
            result_row["model_1_tally_agnostic"] = compute_relative_throughput(tally_agnostic_run_res[job_1], default_job_1_throughput)
            result_row["model_2_tally_agnostic"] = compute_relative_throughput(tally_agnostic_run_res[job_2], default_job_2_throughput)

        co_locate_result.append(result_row)

    if co_locate_result_out:
        co_locate_df = pd.DataFrame(co_locate_result)
        co_locate_df.to_csv(co_locate_result_out, index=True)

    return single_job_result, co_locate_result

if __name__ == "__main__":
    parse_result("result.json")