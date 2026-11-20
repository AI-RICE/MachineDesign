import numpy as np
import pandas as pd

case_info={
    1:{"n_phases":3,"current_n":2,"Rstat":11.4},
    2:{"n_phases":5,"current_n":2,"Rstat":19.0},
    3:{"n_phases":5,"current_n":4,"Rstat":19.0},
}

for case, info in case_info.items:
    case_name=f"{info['n_phases']}f_case{case}"
    data = np.load(f"results/results_HacklGenerator_SixLambdas_{case_name}.npz")
    train_X, train_Y = data["train_X"], data["train_Y"]
    columns = [f"x{i}" for i in range(train_X.shape[1])] + ["loss", "torque", "ripple","voltage"]
    table = pd.DataFrame(np.hstack([train_X, train_Y]), columns=columns)
    table["loss"] = -table["loss"]*info["Rstat"]*(info["n_phases"]/2)
    table["ripple"] = -100 * table["ripple"]
    table.to_csv(f"results/results_HacklGenerator_SixLambdas_{case_name}f.csv", index=False)
