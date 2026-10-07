import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

# parameters
N = 64
L = 10.0
dt = 0.01
steps = 300
theta = 5.0
lam = 10.0

# grid setup
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")

# fourier frequencies
k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX**2 + KY**2

# boundary index shifts
indices = np.arange(N)
shifts = (indices[:, None] - indices[None, :]) % N


# moyal star product in fourier space
def moyal_star_spectral(phi_hat, psi_hat):
    out_hat = np.zeros((N, N), dtype=complex)
    theta_factor = -0.5j * theta

    for i in range(N):
        for j in range(N):
            k_cross_q = KX[i, j] * KY - KY[i, j] * KX
            phase = np.exp(theta_factor * k_cross_q)
            phi_shifted = phi_hat[shifts[i][:, None], shifts[j][None, :]]
            term = phi_shifted * psi_hat * phase
            out_hat[i, j] = np.sum(term)

    return out_hat / (N * N)


# gaussian wave packet
def packet(x0, y0, px, py):
    g = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / 1.0)
    return g * np.cos(px * (X - x0) + py * (Y - y0))


# initial condition with two colliding packets
phi_init = packet(-2.0, 0.0, 4.0, 0.0) + packet(2.0, 0.5, -4.0, 0.0)

phi_c = phi_init.copy()
phi_nc = phi_init.copy()
v_c = np.zeros((N, N))
v_nc = np.zeros((N, N))

history_c = []
history_nc = []

print(
    "Simulation started. N="
    + str(N)
    + " (Warning: Spectral Moyal is slow but accurate)"
)

# initial accelerations
phi_c_hat = np.fft.fft2(phi_c)
a_c_hat = -K2 * phi_c_hat - np.fft.fft2((lam / 6.0) * phi_c**3)
a_c = np.fft.ifft2(a_c_hat).real

phi_nc_hat = np.fft.fft2(phi_nc)
p2_hat = moyal_star_spectral(phi_nc_hat, phi_nc_hat)
p3_hat = moyal_star_spectral(p2_hat, phi_nc_hat)
a_nc_hat = -K2 * phi_nc_hat - (lam / 6.0) * p3_hat
a_nc = np.fft.ifft2(a_nc_hat).real

history_c.append(phi_c.copy())
history_nc.append(phi_nc.copy())

# time evolution loop
for n in range(steps):
    print("Step " + str(n + 1) + "/" + str(steps), end="\r")

    v_c = v_c + 0.5 * dt * a_c
    v_nc = v_nc + 0.5 * dt * a_nc

    phi_c = phi_c + dt * v_c
    phi_nc = phi_nc + dt * v_nc

    phi_c_hat = np.fft.fft2(phi_c)
    a_c_hat = -K2 * phi_c_hat - np.fft.fft2((lam / 6.0) * phi_c**3)
    a_c = np.fft.ifft2(a_c_hat).real

    phi_nc_hat = np.fft.fft2(phi_nc)
    p2_hat = moyal_star_spectral(phi_nc_hat, phi_nc_hat)
    p3_hat = moyal_star_spectral(p2_hat, phi_nc_hat)
    a_nc_hat = -K2 * phi_nc_hat - (lam / 6.0) * p3_hat
    a_nc = np.fft.ifft2(a_nc_hat).real

    v_c = v_c + 0.5 * dt * a_c
    v_nc = v_nc + 0.5 * dt * a_nc

    if n % 2 == 0:
        history_c.append(phi_c.copy())
        history_nc.append(phi_nc.copy())

print("\nDone.")

# plotting and animation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

vmax = np.max(np.abs(history_c)) * 0.8
vmin = -vmax

im1 = ax1.imshow(
    history_c[0].T,
    origin="lower",
    extent=[-L / 2, L / 2, -L / 2, L / 2],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
ax1.set_title("Classical")

im2 = ax2.imshow(
    history_nc[0].T,
    origin="lower",
    extent=[-L / 2, L / 2, -L / 2, L / 2],
    cmap="magma",
    vmin=vmin,
    vmax=vmax,
)
ax2.set_title("Noncommutative (Spectral, theta=" + str(theta) + ")")

im3 = ax3.imshow(
    (history_nc[0] - history_c[0]).T,
    origin="lower",
    extent=[-L / 2, L / 2, -L / 2, L / 2],
    cmap="seismic",
    vmin=-vmax * 0.3,
    vmax=vmax * 0.3,
)
ax3.set_title("Difference")


def update(frame):
    im1.set_data(history_c[frame].T)
    im2.set_data(history_nc[frame].T)
    im3.set_data((history_nc[frame] - history_c[frame]).T)
    return im1, im2, im3


ani = FuncAnimation(
    fig, update, frames=len(history_c), interval=50, blit=False
)
plt.show()