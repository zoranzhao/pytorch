import copy
import argparse
import os
from os import listdir
from os.path import join, isdir

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./data_logs")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument(
    "--benchmark",
    type=str,
    help="torchbench Python script path, e.g. benchmark/dynamo/huggingface.py",
    required=True,
)
parser.add_argument(
    "--model-csv",
    type=str,
    help="""
        the path to csv file to determine the models in the benchmark to collect kernels
        e.g. benchmarks/dynamo/ci_expected_accuracy/inductor_huggingface_training.csv
    """,
    required=True,
)
parser.add_argument("--dtype", type=str, help="delimited list input", default="amp")
parser.add_argument(
    "--mode", type=str, help="delimited list input", default="training,inference"
)


def main(args):
    LOG_DIR = args.log_dir
    DATA_DIR = args.cache_dir
    DTYPE_LIST = args.dtype.split(",")
    MODE_LIST = args.mode.split(",")
    BENCHMARK_PY = args.benchmark
    MODE_LIST_PATH = args.model_csv

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    template = (
        (
            """TORCHINDUCTOR_CACHE_DIR=[[DATA_DIR]]/[[DTYPE]]_[[MODE]]_[[MODEL_NAME]] TORCH_LOGS="+inductor" TORCHINDUCTOR_BENCHMARK_KERNEL=1 python3 [[BENCHMARK_PY]] --[[DTYPE]] --performance --[[MODE]] --inductor -d cuda --filter [[MODEL_NAME]] """
            + " >"
            + LOG_DIR
            + "[[DTYPE]]_[[MODE]]_[[MODEL_NAME]].kernels.log 2>&1"
        )
        .replace("[[DATA_DIR]]", DATA_DIR)
        .replace("[[BENCHMARK_PY]]", BENCHMARK_PY)
    )

    for DTYPE in DTYPE_LIST:
        for MODE in MODE_LIST:
            model_list_path = MODE_LIST_PATH
            model_names = []
            with open(model_list_path, "r") as f:
                for line in f.readlines()[1:]:
                    line = line.split(",")[0]
                    model_names.append(line)
            print(MODE)
            print(model_names)

            for model_name in model_names:
                model_path = "[[DATA_DIR]]/[[DTYPE]]_[[MODE]]_[[MODEL_NAME]]"
                model_path = (
                    model_path.replace("[[MODEL_NAME]]", model_name)
                    .replace("[[MODE]]", MODE)
                    .replace("[[DTYPE]]", DTYPE)
                    .replace("[[DATA_DIR]]", DATA_DIR)
                )

                if isdir(model_path):
                    for kernel in sorted(listdir(model_path)):
                        kernel_path = join(model_path, kernel)
                        if not isdir(kernel_path):
                            continue
                        for file in listdir(kernel_path):
                            file_path = join(kernel_path, file)
                            if file.endswith((".pkl", ".best_config")):
                                cmd = "rm -rf " + file_path
                                print(cmd)
                                os.system(cmd)

                cmd = copy.deepcopy(template)
                cmd = (
                    cmd.replace("[[MODEL_NAME]]", model_name)
                    .replace("[[MODE]]", MODE)
                    .replace("[[DTYPE]]", DTYPE)
                )
                print(cmd)

                os.system(cmd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
