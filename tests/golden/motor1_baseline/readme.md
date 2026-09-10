Golden baseline for motor1 (`motors/synrm_3f_36s.py`), for issue #20.

`tor1.csv` is the raw torque waveform from an Ansys 2025.1 solve of
`motors/synrm_3f_36s.py::Computation` on main.

Setup used:
- generator: HacklGenerator_OneLambda, seed=0
- Nper and PointPer left at their motor defaults (Nper=1/6, PointPer=101)
- r_stator_end=0.7, offset=0.35

Result at capture time: T_mean=4.2946 Nm, n_pts=18.
