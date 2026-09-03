# MachineDesign

This repo contains Python code for automatic rotor design and optimization of synchronous reluctance machines (SynRMs).
The motivation is to improve the performance of SynRMs without using permanent magnets by optimizing the rotor flux barrier geometry. Candidate designs are generated in Python, simulated in Ansys Electronics Desktop, and optimized to increase the average torque and reduce torque ripple.

<img src="figures/result.png" width="600">

Run "run.py" for random design evaluation and "run_optimization.py" for Bayesian optimization. Ansys Electronics Desktop and PyAEDT are required.

## Configuration

Machine-specific settings (Ansys version, number of cores, workers, project directory) are read with 'load_config()' from 'machine_design.config', which merges 'machine_design/config.default.json' with an optional local override at '~/.machine_design/config.json'. Put your own values in the local file instead of editing the defaults or a script directly, so they never end up in a commit.

## Citation

If you use this code, please cite our paper:
[Add citation here]