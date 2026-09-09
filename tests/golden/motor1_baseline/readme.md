Golden baseline for motor1 (`motors/synrm_3f_36s.py`), for issue #20.

`tor.csv` is the raw torque waveform (101 points) from an Ansys 2025.1 solve of
`motors/synrm_3f_36s.py::Computation` on main.

Setup used:
- generator: HacklGenerator_OneLambda, seed=0
- Nper=1 (full electrical period), PointPer=101
- r_stator_end=0.7, offset=0.35

Result at capture time: T_mean=4.3134 Nm, n_pts=101.
