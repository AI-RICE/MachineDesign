import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 1000)


def phase_currents_3f(Id, Iq, m=3):
    Im = np.sqrt(Id**2 + Iq**2)
    epsI = np.arctan2(Iq, Id)
    return [Im * np.cos(theta - np.deg2rad(360 * k / m) + epsI) for k in range(m)]


def phase_currents_5f(Id1, Iq1, Id3, Iq3, m=5):
    Im1 = np.sqrt(Id1**2 + Iq1**2)
    Im3 = np.sqrt(Id3**2 + Iq3**2)
    epsI1 = np.arctan2(Iq1, Id1)
    epsI3 = np.arctan2(Iq3, Id3)
    return [Im1 * np.cos(theta - np.deg2rad(360 * k / m) + epsI1 - np.pi) + Im3 * np.cos(3 * (theta - np.deg2rad(360 * k / m)) + epsI3 - np.pi) for k in range(m)]


currents_3f = phase_currents_3f(Id=5, Iq=5)
currents_5f = phase_currents_5f(Id1=3, Iq1=3, Id3=0, Iq3=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, title, currents in [
    (axes[0], "3f, Id=Iq=5", currents_3f),
    (axes[1], "5f, Id1=Iq1=3, Id3=Iq3=0", currents_5f),
]:
    for k, i_k in enumerate(currents):
        ax.plot(np.degrees(theta), i_k, label=f"phase {'ABCDE'[k]}")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("waveform in phase_currents.png", dpi=600)
plt.show()

for name, is_sq, currents in [
    ("3f", 5**2 + 5**2, currents_3f),
    ("5f", 3**2 + 3**2 + 0**2 + 0**2, currents_5f),
]:
    peak = max(np.max(np.abs(i_k)) for i_k in currents)
    true_loss = sum(np.mean(i_k**2) for i_k in currents)
    print(f"{name}: peak={peak:.3f} A, is^T is={is_sq}, true sum(RMS^2)={true_loss:.3f}, ratio={true_loss / is_sq:.3f}")
