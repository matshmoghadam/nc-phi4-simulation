import numpy as np
import matplotlib.pyplot as plt

# plot grid settings
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 11


# moyal star product implementation
def moyal_star(phi, psi, KX, KY, theta):
    N = phi.shape[0]
    ph = np.fft.fft2(phi)
    ps = np.fft.fft2(psi)
    out_hat = np.zeros((N, N), dtype=complex)
    idx = np.arange(N)

    for i in range(N):
        ii = (i - idx) % N
        for j in range(N):
            jj = (j - idx) % N
            qx = KX[np.ix_(ii, jj)]
            qy = KY[np.ix_(ii, jj)]
            phase = np.exp(-0.5j * theta * (KX * qy - KY * qx))
            out_hat[i, j] = np.sum(ph * ps[np.ix_(ii, jj)] * phase)

    res = np.fft.ifft2(out_hat)
    return res.real / (N**2)


# simulation parameters
N = 64
L = 10.0
theta = 4.0
lam = 2.0
dt = 0.01
steps = 220

dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")

k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX**2 + KY**2

lam_fac = lam / 6.0


# helper functions for derivatives and energy
def laplacian(phi):
    ph = np.fft.fft2(phi)
    return np.fft.ifft2(-K2 * ph).real


def grad_sq(phi):
    ph = np.fft.fft2(phi)
    dphix = np.fft.ifft2(1j * KX * ph).real
    dphiy = np.fft.ifft2(1j * KY * ph).real
    return dphix**2 + dphiy**2


# classical energy
def total_energy_classical(phi, v):
    density = 0.5 * v**2 + 0.5 * grad_sq(phi) + (lam / 24.0) * (phi**4)
    return np.sum(density) * (dx**2)


# noncommutative energy (correct formula with phi_star_phi squared)
def total_energy_nc(phi, v, phi2):
    density = 0.5 * v**2 + 0.5 * grad_sq(phi) + (lam / 24.0) * (phi2**2)
    return np.sum(density) * (dx**2)


def power_spectrum(phi):
    ph = np.fft.fftshift(np.fft.fft2(phi))
    P2 = np.abs(ph) ** 2
    return P2, np.sum(P2)


# initial wave packets
phi_c = np.exp(-((X + 2.2) ** 2 + Y**2) / 1.28) + 0.9 * np.exp(
    -((X - 2.2) ** 2 + (Y - 0.6) ** 2) / 1.28
)
phi_nc = phi_c.copy()

v0 = 0.12 * np.sin(0.9 * Y) * np.exp(-(X**2 + Y**2) / 14.0)
v_c = v0.copy()
v_nc = v0.copy()

# lists to save diagnostic values
max_amp_c = []
max_amp_nc = []
diff_l2 = []
E_c = []
E_nc = []
Pkdiff_sum = []

print(
    "Starting simulation: N="
    + str(N)
    + ", steps="
    + str(steps)
    + ", theta="
    + str(theta)
)

# compute initial accelerations and star product
a_c = laplacian(phi_c) - lam_fac * (phi_c**3)

phi2 = moyal_star(phi_nc, phi_nc, KX, KY, theta)
phi3 = moyal_star(phi2, phi_nc, KX, KY, theta)
a_nc = laplacian(phi_nc) - lam_fac * phi3

# store t=0 initial state metrics
max_amp_c.append(np.max(np.abs(phi_c)))
max_amp_nc.append(np.max(np.abs(phi_nc)))

diff = phi_nc - phi_c
diff_l2.append(np.sqrt(np.mean(diff**2)))

E_c.append(total_energy_classical(phi_c, v_c))
E_nc.append(total_energy_nc(phi_nc, v_nc, phi2))

dummy, Psum = power_spectrum(diff)
Pkdiff_sum.append(Psum)

# time integration with velocity verlet
for n in range(steps):
    # half velocity update
    v_c = v_c + 0.5 * dt * a_c
    v_nc = v_nc + 0.5 * dt * a_nc

    # position update
    phi_c = phi_c + dt * v_c
    phi_nc = phi_nc + dt * v_nc

    # recalculate accelerations
    a_c = laplacian(phi_c) - lam_fac * (phi_c**3)

    phi2 = moyal_star(phi_nc, phi_nc, KX, KY, theta)
    phi3 = moyal_star(phi2, phi_nc, KX, KY, theta)
    a_nc = laplacian(phi_nc) - lam_fac * phi3

    # final half velocity update
    v_c = v_c + 0.5 * dt * a_c
    v_nc = v_nc + 0.5 * dt * a_nc

    # store metrics
    max_amp_c.append(np.max(np.abs(phi_c)))
    max_amp_nc.append(np.max(np.abs(phi_nc)))

    diff = phi_nc - phi_c
    diff_l2.append(np.sqrt(np.mean(diff**2)))

    E_c.append(total_energy_classical(phi_c, v_c))
    E_nc.append(total_energy_nc(phi_nc, v_nc, phi2))

    dummy, Psum = power_spectrum(diff)
    Pkdiff_sum.append(Psum)

print("Simulation complete.")

# plotting output figures
time = np.arange(len(E_c)) * dt

fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

# panel 1: peak amplitude
ax = axs[0, 0]
ax.plot(time, max_amp_c, "-", linewidth=2, label="Classical (C)")
ax.plot(time, max_amp_nc, "--", linewidth=2, label="Noncommutative (NC)")
ax.set_title("Peak amplitude vs time", pad=8)
ax.set_xlabel("Time")
ax.set_ylabel("max |phi|")
ax.legend()

# panel 2: L2 difference
ax = axs[0, 1]
ax.plot(time, diff_l2, "-", linewidth=2)
ax.set_title("Difference growth (L2 norm of NC − C)", pad=8)
ax.set_xlabel("Time")
ax.set_ylabel(r"$\sqrt{\langle(\phi_{NC}-\phi_C)^2\rangle}$")

# panel 3: energy drift
ax = axs[0, 2]
E_c0 = E_c[0]
E_nc0 = E_nc[0]
ax.plot(
    time,
    (np.array(E_c) - E_c0) / abs(E_c0),
    "-",
    linewidth=2,
    label="C energy drift",
)
ax.plot(
    time,
    (np.array(E_nc) - E_nc0) / abs(E_nc0),
    "--",
    linewidth=2,
    label="NC energy drift",
)
ax.set_title("Energy drift (relative)", pad=8)
ax.set_xlabel("Time")
ax.set_ylabel(r"$(E(t)-E(0))/|E(0)|$")
ax.legend()

# panel 4: final cross section
ax = axs[1, 0]
mid_idx = N // 2
ax.plot(x, phi_c[:, mid_idx], "-", linewidth=2, label="C final")
ax.plot(x, phi_nc[:, mid_idx], "--", linewidth=2, label="NC final")
ax.plot(x, (phi_nc - phi_c)[:, mid_idx], ":", linewidth=2, label="NC − C final")
ax.set_title("Final cross-section at y=0 (middle row)", pad=8)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.legend()

# panel 5: spectral diff map
ax5 = axs[1, 1]
diff_field = phi_nc - phi_c
diff_fft = np.fft.fftshift(np.abs(np.fft.fft2(diff_field)))
log_map = np.log10(diff_fft + 1e-10)

im = ax5.imshow(log_map, origin="lower", cmap="viridis", vmin=-10, vmax=0)
ax5.set_title("log10 |FFT(NC − C)| (shifted)", pad=8)
ax5.set_xlabel("k-index")
ax5.set_ylabel("k-index")

fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

# panel 6: difference spectral power
ax = axs[1, 2]
ax.plot(time, Pkdiff_sum, "-", linewidth=2)
ax.set_title("Total spectral power of difference", pad=8)
ax.set_xlabel("Time")
ax.set_ylabel(r"$\sum_k |\Delta\phi_k|^2$")

plt.show()