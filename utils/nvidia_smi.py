import json
import os
import subprocess
import time
import traceback
import pandas as pd

from xml.dom import minidom


def parse_smi_list(smi_list):

    smi_df = pd.DataFrame(list(smi_list))
    smi_df.drop([0])
    smi_df.drop([len(smi_df)-1], inplace=True)

    gpu_util = float(round(pd.to_numeric(smi_df['gpuUtil']).mean(), 3))
    gmem_util = float(round(pd.to_numeric(smi_df['gpuMemUtil']).mean(), 3))
    gmem = float(round(pd.to_numeric(smi_df['gpuMem']).max(), 3))

    return  {
        'gpu_util': gpu_util,
        'gmem_util': gmem_util,
        'gmem': gmem
    }


def smi_getter(smi_list, gpu_id):
    metrics_output_dir = "./"
    if len(gpu_id) == 1:
        cmd = f"nvidia-smi -q -x -i {gpu_id[0]}".split()
    elif len(gpu_id) == 2:
        cmd = f"nvidia-smi -q -x -i {gpu_id[0]},{gpu_id[1]}".split()
    elif len(gpu_id) == 4:
        cmd = f"nvidia-smi -q -x -i {gpu_id[0]},{gpu_id[1]},{gpu_id[2]},{gpu_id[3]}".split()
    while True:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            smi_output = p.stdout.read()
        except Exception:
            traceback.print_exc()
            gen_empty_gpu_metric(metrics_output_dir)
        output = parse_nvidia_smi_result(smi_output, metrics_output_dir, gpu_id)
        smi_list.extend(output)
        # TODO: change to sleep time configurable via arguments
        time.sleep(0.2)


def parse_nvidia_smi_result(smi, outputDir, gpu_id):
    try:
        old_umask = os.umask(0)
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName("gpu")
        gpuInfo = []
        outPut = {}
        outPut["Timestamp"] = time.asctime(time.localtime())
        for gpuIndex, gpu in enumerate(gpuList):
            outPut["index"] = gpu_id[gpuIndex]
            outPut["gpuUtil"] = (
                gpu.getElementsByTagName("utilization")[0]
                .getElementsByTagName("gpu_util")[0]
                .childNodes[0]
                .data.replace("%", "")
                .strip()
            )
            outPut["gpuMemUtil"] = (
                gpu.getElementsByTagName("utilization")[0]
                .getElementsByTagName("memory_util")[0]
                .childNodes[0]
                .data.replace("%", "")
                .strip()
            )
            outPut["gpuMem"] = (
                gpu.getElementsByTagName("fb_memory_usage")[0]
                .getElementsByTagName("used")[0]
                .childNodes[0]
                .data
            ).strip("MiB").strip()
            # processes = gpu.getElementsByTagName("processes")
            # runningProNumber = len(processes[0].getElementsByTagName("process_info"))
            # gpuInfo["activeProcessNum"] = runningProNumber

            # print(outPut)
            gpuInfo.append(outPut.copy())
        return gpuInfo

    except Exception as error:
        # e_info = sys.exc_info()
        print("gpu_metrics_collector error: %s" % error)
    finally:
        os.umask(old_umask)


def gen_empty_gpu_metric(outputDir):
    try:
        old_umask = os.umask(0)
        with open(os.path.join(outputDir, "gpu_metrics"), "a") as outputFile:
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = 0
            outPut["gpuInfos"] = []
            print(outPut)
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    except Exception:
        traceback.print_exc()
    finally:
        os.umask(old_umask)

