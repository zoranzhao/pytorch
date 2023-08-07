import pickle
import argparse
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, default="../../raw_data.pkl")

raw_data_file = parser.parse_args().raw_data
with open(raw_data_file, "rb") as file:
    raw_data = pickle.load(file)

"""
{'kernel_counter': 0, 'model': 'amp_inference_AlbertForMaskedLM', 'kernel_name': 'c4g7tlkaqouyvjoqhzzrc5kyswlg2zzf3dnmumg3u6fuzqb2d2vq', 'kernel_category': 1, 'num_of_loops': 3, 'op_bag': {2: 5, 4: 3, 5: 2, 0: 12, 10: 2, 3: 2, 9: 1, 8: 1, 6: 2, 7: 2, 1: 4}, 'size_hints': [2048, 4096], 'reads_list': [{'StarDepOrWeakDep': False, 'bytes': 16384, 'strides': [0, 1], 'size': [2048, 4096], 'is_contiguous': True, 'is_scalar': False, 'is_indirect': False, 'name': 'arg21_1'}, {'StarDepOrWeakDep': False, 'bytes': 16384, 'strides': [0, 1], 'size': [2048, 4096], 'is_contiguous': True, 'is_scalar': False, 'is_indirect': False, 'name': 'arg22_1'}, {'StarDepOrWeakDep': False, 'bytes': 33554432, 'strides': [1], 'size': [8388608], 'is_contiguous': True, 'is_scalar': False, 'is_indirect': False, 'name': 'buf472'}, {'StarDepOrWeakDep': False, 'bytes': 16777216, 'strides': [1], 'size': [8388608], 'is_contiguous': True, 'is_scalar': False, 'is_indirect': False, 'name': 'buf480'}], 'writes_list': [{'StarDepOrWeakDep': False, 'bytes': 16777216, 'strides': [1], 'size': [8388608], 'is_contiguous': True, 'is_scalar': False, 'is_indirect': False, 'name': 'buf484'}], 'config': {'XBLOCK': 1, 'RBLOCK': 512, 'num_warps': 16, 'num_stages': 1, 'n_regs': 30, 'n_spills': 0, 'shared': 2048, 'timing': 0.07372800260782242}}
"""

### feature vector
# qid: query id -> kernel_counter: int
# kernel_category: int
# num_of_loops: int
# op_bag: [int]
# size_hints: [int]
# reads [[int]]
#  - StarDepOrWeakDep: bool
#  - bytes: int
#  - strides: [int]
#  - size: [int]
#  - is_contiguous: bool
#  - is_scalar: bool
#  - is_indirect: bool
# writes [[int]]
#  - StarDepOrWeakDep: bool
#  - bytes: int
#  - strides: [int]
#  - size: [int]
#  - is_contiguous: bool
#  - is_scalar: bool
#  - is_indirect: bool
# config
#  - XBLOCK: int
#  - RBLOCK/YBLOCK: int
#  - num_warps: int
#  - num_stages: int
#  - n_regs: int (removed)
#  - n_spills: int (removed)
#  - shared: int (removed)

### label
# timing: float


X = list()
y = list()
y_baseline = list()
qid = list()

op_bag_len = 0
size_hints_len = 0
strides_len = 0
size_len = 0
reads_len = 0
writes_len = 0


def show_feature(v):
    print("kernel_category", v[0])
    print("num_of_loops", v[1])
    print("op_bag", v[2:58])
    print("size_hints", v[58:60])
    for i in range(10):
        print(f"reads[{i}]", v[60 + i * 17 : 60 + i * 17 + 17])
    for i in range(5):
        print(f"writes[{i}]", v[230 + i * 17 : 230 + i * 17 + 17])
    print("XBLOCK", v[315])
    print("YBLOCK", v[316])
    print("RBLOCK", v[317])
    print("num_warps", v[318])
    print("num_stages", v[319])
    # print("num_regs", v[319])
    # print("num_spills", v[320])
    # print("num_shared", v[321])
    print("xnumel", v[320])
    print("ynumel", v[321])
    print("rnumel", v[322])


for config in raw_data:
    op_bag_len = max(op_bag_len, max(config["op_bag"].keys()) + 1)
    size_hints_len = max(size_hints_len, len(config["size_hints"]))
    reads_len = max(reads_len, len(config["reads_list"]))
    writes_len = max(writes_len, len(config["writes_list"]))
    for read in config["reads_list"]:
        strides_len = max(strides_len, len(read["strides"]))
        size_len = max(size_len, len(read["size"]))
    for write in config["writes_list"]:
        strides_len = max(strides_len, len(write["strides"]))
        size_len = max(size_len, len(write["size"]))

print(op_bag_len, size_hints_len, reads_len, writes_len, strides_len, size_len)


def pad_tensor():
    tensor_feature = list()
    tensor_feature.append(False)  # StarDepOrWeakDep
    tensor_feature.append(0)  # bytes
    tensor_feature.extend([0] * strides_len)  # strides
    tensor_feature.extend([0] * size_len)  # size
    tensor_feature.append(True)  # is_contiguous
    tensor_feature.append(False)  # is_scalar
    tensor_feature.append(False)  # is_indirect
    return tensor_feature


def tensor_list(rw_list, rw_len):
    res = list()
    rw_list = sorted(rw_list, key=lambda x: x["bytes"], reverse=True)
    for tensor in rw_list[:rw_len]:
        tensor_feature = pad_tensor()
        tensor_feature[0] = tensor["StarDepOrWeakDep"]
        tensor_feature[1] = tensor["bytes"]
        # left pad strides
        for i in range(len(tensor["strides"])):
            tensor_feature[8 - (len(tensor["strides"]) - i)] = tensor["strides"][i]
        # left pad size
        for i in range(len(tensor["size"])):
            tensor_feature[14 - (len(tensor["size"]) - i)] = tensor["size"][i]
        tensor_feature[-3] = tensor["is_contiguous"]
        tensor_feature[-2] = tensor["is_scalar"]
        tensor_feature[-1] = tensor["is_indirect"]
        res.append(tensor_feature)
    for i in range(rw_len - len(rw_list)):
        res.append(pad_tensor())
    return res


reads_len = 10
writes_len = 5

np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=30, linewidth=100000)

for config in raw_data:
    feature_vector = list()
    feature_vector.append(config["kernel_category"])
    feature_vector.append(config["num_of_loops"])
    op_bag = [0] * op_bag_len
    for op in config["op_bag"]:
        op_bag[op] = config["op_bag"][op]
    feature_vector.extend(op_bag)
    size_hints = [1] * size_hints_len
    for i in range(len(config["size_hints"])):
        size_hints[i] = config["size_hints"][i]
    feature_vector.extend(size_hints)
    reads = tensor_list(config["reads_list"], reads_len)
    for tensor in reads:
        feature_vector.extend(tensor)
    writes = tensor_list(config["writes_list"], writes_len)
    for tensor in writes:
        feature_vector.extend(tensor)

    if "XBLOCK" in config["config"]:
        feature_vector.append(config["config"]["XBLOCK"])
    else:
        feature_vector.append(1)
    if "YBLOCK" in config["config"]:
        feature_vector.append(config["config"]["YBLOCK"])
    else:
        feature_vector.append(1)
    if "RBLOCK" in config["config"]:
        feature_vector.append(config["config"]["RBLOCK"])
    else:
        feature_vector.append(1)

    feature_vector.append(config["config"]["num_warps"])
    feature_vector.append(config["config"]["num_stages"])
    # feature_vector.append(config["config"]["n_regs"])
    # feature_vector.append(config["config"]["n_spills"])
    # feature_vector.append(config["config"]["shared"])
    feature_vector.append(config["tiling"][0])
    feature_vector.append(config["tiling"][1])
    feature_vector.append(config["tiling"][2])

    if np.isinf(config["config"]["timing"]):
        config["config"]["timing"] = 1e6

    X.append(feature_vector)
    y.append(config["config"]["timing"])
    y_baseline.append(config["baseline_timing"])
    qid.append(config["kernel_counter"])

X = np.array(X)
y = np.array(y)
y_baseline = np.array(y_baseline)
qid = np.array(qid)

with open("X.pkl", "wb") as f:
    pickle.dump(X, f)
with open("y.pkl", "wb") as f:
    pickle.dump(y, f)
with open("y_baseline.pkl", "wb") as f:
    pickle.dump(y_baseline, f)
with open("qid.pkl", "wb") as f:
    pickle.dump(qid, f)

show_feature(X[0])
print(y[0])
print(y_baseline[0])
print(qid[0])

assert not np.any(np.isnan(X))
assert not np.any(np.isinf(X))

X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
show_feature(X_max)
show_feature(X_min)

y_max = np.max(y, axis=0)
y_min = np.min(y, axis=0)
print(y_max)
print(y_min)

print(X.shape)
print(y.shape)
print(y_baseline.shape)
print(qid.shape)

train_indices, test_indices = next(
    GroupShuffleSplit(
        train_size=0.80, test_size=0.20, n_splits=1, random_state=0
    ).split(X, y, qid)
)

X_train = X[train_indices]
y_train = y[train_indices]
y_baseline_train = y_baseline[train_indices]
qid_train = qid[train_indices]

sorted_indices = np.argsort(qid_train)
X_train = X_train[sorted_indices]
y_train = y_train[sorted_indices]
y_baseline_train = y_baseline_train[sorted_indices]
qid_train = qid_train[sorted_indices]

y_train_normalized = y_train.copy()
for train_id in np.unique(qid_train):
    y_min = np.min(y_train[qid_train == train_id])
    y_train_normalized[qid_train == train_id] = y_min / y_train[qid_train == train_id]

print(X_train.shape)
print(y_train.shape)
print(qid_train.shape)
qid_train_unique = np.unique(qid_train)
print(qid_train_unique[:10])


X_test = X[test_indices]
y_test = y[test_indices]
y_baseline_test = y_baseline[test_indices]
qid_test = qid[test_indices]

sorted_indices = np.argsort(qid_test)
X_test = X_test[sorted_indices]
y_test = y_test[sorted_indices]
y_baseline_test = y_baseline_test[sorted_indices]
qid_test = qid_test[sorted_indices]

y_test_normalized = y_test.copy()
for test_id in np.unique(qid_test):
    y_min = np.min(y_test[qid_test == test_id])
    y_test_normalized[qid_test == test_id] = y_min / y_test[qid_test == test_id]

print(X_test.shape)
print(y_test.shape)
print(qid_test.shape)
qid_test_unique = np.unique(qid_test)
print(qid_test_unique[:10])

with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("y_train_unnormalized.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train_normalized, f)
with open("y_baseline_train.pkl", "wb") as f:
    pickle.dump(y_baseline_train, f)
with open("qid_train.pkl", "wb") as f:
    pickle.dump(qid_train, f)

with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("y_test_unnormalized.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test_normalized, f)
with open("y_baseline_test.pkl", "wb") as f:
    pickle.dump(y_baseline_test, f)
with open("qid_test.pkl", "wb") as f:
    pickle.dump(qid_test, f)
