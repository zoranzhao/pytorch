import copy
import os
from os import listdir
from os.path import isfile, join, isdir

LOG_DIR = "./data-logs-hf/"

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

template = (
    """TORCHINDUCTOR_CACHE_DIR=/scratch/bohanhou/fresh/data_hf/[[DTYPE]]_[[MODE]]_[[MODEL_NAME]] TORCH_LOGS="+inductor" TORCHINDUCTOR_BENCHMARK_KERNEL=1 python3 benchmarks/dynamo/huggingface.py --[[DTYPE]] --performance --[[MODE]] --inductor -d cuda --filter [[MODEL_NAME]] """
    + " >"
    + LOG_DIR
    + "[[DTYPE]]_[[MODE]]_[[MODEL_NAME]].kernels.log 2>&1"
)

for DTYPE in ["amp"]:
    for MODE in ["training", "inference"]:
        model_list_path = "/scratch/bohanhou/fresh/pytorch/benchmarks/dynamo/ci_expected_accuracy/inductor_huggingface_[[MODE]].csv"
        model_list_path = model_list_path.replace("[[MODE]]", MODE)
        model_names = []
        with open(model_list_path, "r") as f:
            for line in f.readlines()[1:]:
                line = line.split(",")[0]
                model_names.append(line)
        print(MODE)
        print(model_names)

        for model_name in model_names:
            model_path = (
                "/scratch/bohanhou/fresh/data_hf/[[DTYPE]]_[[MODE]]_[[MODEL_NAME]]"
            )
            model_path = (
                model_path.replace("[[MODEL_NAME]]", model_name)
                .replace("[[MODE]]", MODE)
                .replace("[[DTYPE]]", DTYPE)
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
