Golden baseline for motor2 (now `motors/synrm_5f_40s.py`), for issue #20.

`tor.csv` is the raw torque waveform (101 points) from Ansys 2025.1 solve of
`machine_design/design2.py::Design2` at commit `c438e18` (dropped the unsupported
deriv() voltage terms), the earliest version confirmed to be free of known bugs
(Rstat, set_derived_params, deriv()) at the time of writing.

Reproduce with (checked out at commit c438e18, with Ansys 2025.1 installed):

    python notebooks/run2.py --nper 1 --aedt-version 2025.1

Setup used:
- generator: HacklGenerator_OneLambda, seed=0
- current setpoint: Id1=7.0711, Iq1=7.0711, Id3=0.0, Iq3=0.0
- Nper=1 (full electrical period), PointPer=101
- r_stator_end=0.7, offset=0.35

Result at capture time: T_mean=35.6431 Nm, ripple=11.374 %, n_pts=101.
