Golden baseline for motor2 (`motors/synrm_5f_40s.py`), for issue #20.

`tor.csv` is the raw torque waveform (101 points) from an Ansys 2025.1 solve of
`motors/synrm_5f_40s.py::Computation` on main.

Setup used:
- generator: HacklGenerator_OneLambda, seed=0
- current setpoint: Id1=7.0711, Iq1=7.0711, Id3=0.0, Iq3=0.0
- Nper=1 (full electrical period), PointPer=101
- r_stator_end=0.7, offset=0.35

Result at capture time: T_mean=35.6560 Nm, n_pts=101.
