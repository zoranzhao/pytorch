KERNEL_DIR = "./data_tb"

import subprocess, os, signal
from os import listdir
from os.path import isfile, join, isdir

seen_kernels = set()

for model in sorted(listdir(KERNEL_DIR)):
    model_path = join(KERNEL_DIR, model)
    if not isdir(model_path):
        continue

    for kernel in sorted(listdir(model_path)):
        kernel_path = join(model_path, kernel)
        if not isdir(kernel_path):
            continue

        # remove best config file
        for py in listdir(kernel_path):
            py_path = join(kernel_path, py)
            if py.endswith((".best_config")):
                cmd = "rm -rf " + py_path
                print(cmd)
                os.system(cmd)

        # run kernel
        for py in listdir(kernel_path):
            py_path = join(kernel_path, py)
            if not py.endswith(".py"):
                continue

            # skip graph python file
            with open(py_path, "r") as file:
                content = file.read()
                if "Original ATen:" in content:
                    print("Skip " + py_path + " GRAPH")
                    continue

            # skip seen kernels
            if py[:-3] in seen_kernels:
                print("Skip " + py_path + " <<<<<< " + py[:-3] + " seen before")
                continue

            cache_dir = kernel_path
            log_path = join(kernel_path, py[:-3] + ".log")
            all_config_path = join(kernel_path, py[:-3] + ".all_config")

            if os.path.exists(log_path) and os.path.exists(all_config_path):
                # already benchmarked
                seen_kernels.add(py[:-3])
                continue
            assert not os.path.exists(log_path) and not os.path.exists(all_config_path)
            
            my_env = os.environ.copy()
            my_env["TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"] = "1"
            my_env["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
            my_env["TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS"] = "2"
            my_env["TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS"] = "1"
            my_env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            my_env["TORCH_LOGS"] = "+inductor"
            my_env["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
            cmd = """python3 [[PY_PATH]] > [[LOG_PATH]] 2>&1"""
            cmd = (
                cmd.replace("[[CACHE_DIR]]", cache_dir)
                .replace("[[PY_PATH]]", py_path)
                .replace("[[LOG_PATH]]", log_path)
            )
            print(cmd)
            try:
                pro = subprocess.Popen(
                    cmd, env=my_env, shell=True, preexec_fn=os.setsid
                )
                pro.wait(timeout=90)
            except subprocess.TimeoutExpired as exc:
                print(exc)
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

            seen_kernels.add(py[:-3])
