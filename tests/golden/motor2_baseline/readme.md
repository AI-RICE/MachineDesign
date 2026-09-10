Golden baseline for motor2 (`motors/synrm_5f_40s.py`), for issue #20.

`tor2.csv` is the raw torque waveform from an Ansys 2025.1 solve of
`motors/synrm_5f_40s.py::Computation` on main.

Setup used:
- generator: HacklGenerator_OneLambda, seed=0
- current setpoint: Id1=7.0711, Iq1=7.0711, Id3=0.0, Iq3=0.0
- Nper and PointPer left at their motor defaults (Nper=1/10, PointPer=101)
- r_stator_end=0.7, offset=0.35

Result at capture time: T_mean=35.1976 Nm, n_pts=11.
