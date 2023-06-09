import xgboost
import pickle
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

np.random.seed(0)
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(edgeitems=30, linewidth=100000)


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


with open("X_train.pkl", "rb") as file:
    X_train = pickle.load(file)
with open("y_train.pkl", "rb") as file:
    y_train = pickle.load(file)
with open("qid_train.pkl", "rb") as file:
    qid_train = pickle.load(file)

with open("X_test.pkl", "rb") as file:
    X_test = pickle.load(file)
with open("y_test.pkl", "rb") as file:
    y_test = pickle.load(file)
with open("y_test_unnormalized.pkl", "rb") as file:
    y_test_unnormalized = pickle.load(file)
with open("y_baseline_test.pkl", "rb") as file:
    y_baseline_test = pickle.load(file)
with open("qid_test.pkl", "rb") as file:
    qid_test = pickle.load(file)


# model = xgboost.XGBRanker()
# model.load_model("model.bin")
model = xgboost.XGBRegressor()
model.load_model("model_normalize_runtime_all_cpu.bin")

qid_test_unique = np.unique(qid_test)
print(qid_test_unique[:10])

avg_rel_top1 = 0
avg_rel_top2 = 0
avg_rel_top3 = 0
avg_rel_err_true = 0

acc_top1 = 0
acc_top2 = 0
acc_top3 = 0

import time

for i, test_id in enumerate(qid_test_unique):
    time_start = time.time()    
    scores = model.predict(X_test[qid_test == test_id])
    time_end = time.time()
    if i < 20:
        print(
            f"test_id: {test_id}, time: {time_end - time_start}, len: {X_test[qid_test == test_id].shape[0]}"
        )

    y_pred_arr = y_test_unnormalized[qid_test == test_id][np.argsort(scores)[::-1]]
    y_pred_top1 = y_pred_arr[:1].min()
    y_pred_top2 = y_pred_arr[:2].min()
    y_pred_top3 = y_pred_arr[:5].min()

    y_pred = y_pred_arr[y_pred_arr != 1e6][0]

    y_baseline = y_baseline_test[qid_test == test_id][0]
    y_true = y_test_unnormalized[qid_test == test_id].min()

    acc_top1 += y_pred_top1 == y_true
    acc_top2 += y_pred_top2 == y_true
    acc_top3 += y_pred_top3 == y_true

    avg_rel_top1 += (y_pred_top1 - y_baseline) / y_baseline
    avg_rel_top2 += (y_pred_top2 - y_baseline) / y_baseline
    avg_rel_top3 += (y_pred_top3 - y_baseline) / y_baseline
    avg_rel_err_true += (y_true - y_baseline) / y_baseline

    # print(
    #     f"test_id: {test_id}, y_pred: {y_pred}, y_true: {y_true} y_baseline: {y_baseline}\n",
    #     avg_rel_top1 / (i + 1) * 100,
    #     avg_rel_top2 / (i + 1) * 100,
    #     avg_rel_top3 / (i + 1) * 100,
    #     avg_rel_err_true / (i + 1) * 100,
    #     # acc_top1 / (i + 1) * 100,
    #     # acc_top2 / (i + 1) * 100,
    #     # acc_top3 / (i + 1) * 100,
    # )

print("acc_top1", acc_top1 / len(qid_test_unique) * 100)
print("acc_top2", acc_top2 / len(qid_test_unique) * 100)
print("acc_top3", acc_top3 / len(qid_test_unique) * 100)
print("avg_rel_top1", avg_rel_top1 / len(qid_test_unique) * 100)
print("avg_rel_top2", avg_rel_top2 / len(qid_test_unique) * 100)
print("avg_rel_top3", avg_rel_top3 / len(qid_test_unique) * 100)
print("avg_rel_err_true", avg_rel_err_true / len(qid_test_unique) * 100)


def get_loss(X_loss, y_loss):
    device = "cuda"
    batch_size = 4096
    mse_loss_sum = 0
    mae_loss_sum = 0

    with torch.no_grad():
        for i in range(0, X_loss.shape[0], batch_size):
            y_pred = torch.from_numpy(model.predict(X_loss[i : i + batch_size])).to(
                device
            )
            mse_loss = torch.nn.functional.mse_loss(
                y_pred.squeeze(),
                torch.from_numpy(y_loss[i : i + batch_size]).to(device),
            )
            mae_loss = torch.nn.functional.l1_loss(
                y_pred.squeeze(),
                torch.from_numpy(y_loss[i : i + batch_size]).to(device),
            )

            mse_loss_sum += mse_loss.item() * y_pred.shape[0]
            mae_loss_sum += mae_loss.item() * y_pred.shape[0]
        torch.cuda.empty_cache()

    return mse_loss_sum / X_loss.shape[0], mae_loss_sum / X_loss.shape[0]


mse_loss, mae_loss = get_loss(X_train, y_train)
print(
    f"Train:  rmse_loss={np.sqrt(mse_loss)} mae_loss={mae_loss}",
    end=" ||| ",
)

mse_loss, mae_loss = get_loss(X_test, y_test)
print(f"Test: rmse_loss={np.sqrt(mse_loss)} mae_loss={mae_loss}")

# dump feature importance
print(np.sort(model.feature_importances_))
sorted_indices = np.flip(np.argsort(model.feature_importances_))
print(sorted_indices)
print(len(sorted_indices))
