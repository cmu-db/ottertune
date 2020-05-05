import json
import sys
import copy
import argparse
import os
sys.path.append("../../../")
sys.path.append("../")
from server.website.website.types import VarType   # pylint: disable=import-error,wrong-import-position,line-too-long  # noqa: E402
from driver_config import OLTPBENCH_HOME  # pylint: disable=import-error,wrong-import-position  # noqa: E402

parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument("result_dir")
args = parser.parse_args()  # pylint: disable=invalid-name

USER_DEINFED_METRICS = {
    "throughput": {
        "unit": "transaction / second",
        "short_unit": "txn/s",
        "type": VarType.INTEGER
    },
    "latency_99": {
        "unit": "microseconds",
        "short_unit": "us",
        "type": VarType.INTEGER
    },
    "latency_95": {
        "unit": "microseconds",
        "short_unit": "us",
        "type": VarType.INTEGER
    }
}


def get_udm():
    summary_path = OLTPBENCH_HOME + '/results/outputfile.summary'
    with open(summary_path, 'r') as f:
        info = json.load(f)
    metrics = copy.deepcopy(USER_DEINFED_METRICS)
    metrics["throughput"]["value"] = info["Throughput (requests/second)"]
    metrics["latency_99"]["value"] =\
        info["Latency Distribution"]["99th Percentile Latency (microseconds)"]
    metrics["latency_95"]["value"] =\
        info["Latency Distribution"]["95th Percentile Latency (microseconds)"]
    return metrics


def write_udm():
    metrics = get_udm()
    result_dir = args.result_dir
    path = os.path.join(result_dir, 'user_defined_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    write_udm()
