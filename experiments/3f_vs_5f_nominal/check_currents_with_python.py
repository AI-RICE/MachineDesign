import numpy as np
import matplotlib.pyplot as plt

from machine_design.generic.transforms import to_phases

x = 1.5
theta_el = np.linspace(-np.pi, np.pi, 361)

I_3f = to_phases(x, x, theta_el, harmonic=1, n_phases=3)
I_5f = to_phases(3 / 5 * x, 3 / 5 * x, theta_el, harmonic=1, n_phases=5)

Rstat_3f = 1.0
Rstat_5f = 19.0

V_3f = Rstat_3f * I_3f
V_5f = Rstat_5f * I_5f

fig, axes = plt.subplots(2, 2, sharex=True)
axes[0, 0].plot(theta_el, I_3f)
axes[0, 0].set_title(f"3f currents, (Id1,Iq1)=({x},{x})")
axes[0, 1].plot(theta_el, I_5f)
axes[0, 1].set_title(f"5f currents, (Id1,Iq1)=(3/5*{x},3/5*{x}),Id3=Iq3=0")
axes[1, 0].plot(theta_el, V_3f)
axes[1, 0].set_title("3f V=Rstat*I(Rstat=1.0)")
axes[1, 1].plot(theta_el, V_5f)
axes[1, 1].set_title("5f V=Rstat*I(Rstat=19.0)")
plt.tight_layout()
plt.savefig("currents_and_voltage.png")
plt.show()
