import xgboost
import pickle
import numpy as np
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


with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("qid_train.pkl", "rb") as f:
    qid_train = pickle.load(f)

with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("qid_test.pkl", "rb") as f:
    qid_test = pickle.load(f)

qid_train_unique = np.unique(qid_train)
print(qid_train_unique[:10])

qid_test_unique = np.unique(qid_test)
print(qid_test_unique[:10])

assert np.intersect1d(qid_train, qid_test).size == 0

ranker = xgboost.XGBRegressor(
    max_depth=15,
    learning_rate=0.2,
    n_estimators=500,
    tree_method="hist",
    predictor="cpu_predictor",
    eval_metric=["rmse", "mae"],
)

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

ranker.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True,
)

qid_test_unique = np.unique(qid_test)
test_id = qid_test_unique[1]

print(test_id)
scores = ranker.predict(X_test[qid_test == test_id])

indices = np.argsort(scores)[::-1]
print(scores[indices])
print(y_test[qid_test == test_id][indices])

# dump model
ranker.save_model("model_normalize_runtime_all_cpu.bin")
