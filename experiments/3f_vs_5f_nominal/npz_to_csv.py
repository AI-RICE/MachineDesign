import numpy as np
import pandas as pd

for n_phases in [3, 5]:
    data = np.load(f"results/results_HacklGenerator_SixLambdas_{n_phases}f.npz")
    train_X, train_Y = data["train_X"], data["train_Y"]
    columns = [f"x{i}" for i in range(train_X.shape[1])] + ["loss", "torque", "ripple"]
    table = pd.DataFrame(np.hstack([train_X, train_Y]), columns=columns)
    table.to_csv(f"results/results_HacklGenerator_SixLambdas_{n_phases}f.csv", index=False)
