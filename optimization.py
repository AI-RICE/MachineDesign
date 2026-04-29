import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

metadata = pd.read_csv("results/metadata.csv")

files = glob.glob("results/*.pkl")

all_barriers = []
all_torque = []
y_mins = []
w_mins = []
y_mids = []
w_mids = []
thetas = []
w_maxs = []
ripples = []

for file in files:
    fname = os.path.basename(file).replace(".pkl", "")
    parts = fname.split("_")
    method = parts[1]
    design = int(parts[2])
    barriers = int(parts[4])

    row = metadata[(metadata['method'] == method) &
                   (metadata['design'] == design) &
                   (metadata['n_barriers'] == barriers)]
    if row.empty:
        print(f"Metadata nenalezena pro {fname}")
        continue

    T = float(row['T'])
    ripple = float(row['ripple'])

    with open(file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        y_mins_ = data.get("y_mins")
        w_mins_ = data.get("w_mins")
        y_mids_ = data.get("y_mids")
        w_mids_ = data.get("w_mids")
        thetas_ = data.get("thetas")
        w_maxs_ = data.get("w_maxs")
    elif isinstance(data, tuple):
        y_mins_, w_mins_, y_mids_, w_mids_, thetas_, w_maxs_ = data
    else:
        continue

    all_barriers.append(barriers)
    all_torque.append(T)
    ripples.append(ripple)
    y_mins.append(np.mean(y_mins_))
    w_mins.append(np.mean(w_mins_))
    y_mids.append(np.mean(y_mids_))
    w_mids.append(np.mean(w_mids_))
    thetas.append(np.mean(thetas_))
    w_maxs.append(np.mean(w_maxs_))

all_barriers = np.array(all_barriers)
all_torque = np.array(all_torque)
ripples = np.array(ripples)

params = {
    "w_mins": np.array(w_mins),
    "y_mins": np.array(y_mins),
    "y_mids": np.array(y_mids),
    "w_mids": np.array(w_mids),
    "thetas": np.array(thetas),
    "w_maxs": np.array(w_maxs),
    "ripple": np.array(ripples),
}

high_corr_names = []
for name, values in params.items():
    if np.std(values) == 0:
        continue
    corr = np.corrcoef(values, all_torque)[0, 1]
    print(f"{name}: {corr:.3f}")
    if not np.isnan(corr) and abs(corr) > 0.5:
        high_corr_names.append(name)

print("Pouzite parametry pro optimalizaci:", high_corr_names)

if len(high_corr_names) == 0:
    raise ValueError("Žádné parametry nemají dostatečnou korelaci.")

X = np.column_stack([params[name] for name in high_corr_names])
Y = all_torque

interpolator = Rbf(*[X[:, i] for i in range(X.shape[1])], Y, function="linear")

def objective(p):
    val = float(interpolator(*p))
    if np.isnan(val):
        return np.inf
    return -val

x0 = X[np.argmax(Y)]
res = minimize(objective, x0, method="Nelder-Mead")

best_barriers = None
best_torque = -np.inf

for i in np.unique(all_barriers):
    indices = np.where(all_barriers == i)[0]
    if len(indices) == 0:
        continue
    torque_for_i = np.max(all_torque[indices])
    if torque_for_i > best_torque:
        best_torque = torque_for_i
        best_barriers = i

print("Optimalni parametry:", res.x)
print("Maximalni moment:", -res.fun)
print("Optimalni pocet barier:", best_barriers)

barrier_values = sorted(np.unique(all_barriers))

data_to_plot = []
for n in barrier_values:
    idx = np.where(all_barriers == n)[0]
    data_to_plot.append(all_torque[idx])

plt.figure(figsize=(8,5))
plt.boxplot(data_to_plot, tick_labels=barrier_values)
plt.xlabel("Počet bariér")
plt.ylabel("Moment (T)")
plt.title("Rozptyl momentu podle počtu bariér")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()