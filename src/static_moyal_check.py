import numpy as np
import matplotlib.pyplot as plt

# function to compute moyal star product of two fields
def moyal_star(phi_a, phi_b, KX, KY, theta):
    N = phi_a.shape[0]

    # transform to momentum space
    A_k = np.fft.fft2(phi_a)
    B_k = np.fft.fft2(phi_b)

    C_k = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            kx = KX[i, j]
            ky = KY[i, j]

            sum_k = 0.0 + 0.0j
            for m in range(N):
                for n in range(N):
                    px = KX[m, n]
                    py = KY[m, n]

                    # periodic boundary condition
                    jx = i - m
                    if jx < 0:
                        jx = jx + N

                    jy = j - n
                    if jy < 0:
                        jy = jy + N

                    # calculate phase
                    phase_angle = 0.5 * theta * (px * ky - py * kx)
                    phase = np.exp(1j * phase_angle)

                    sum_k = sum_k + A_k[m, n] * B_k[jx, jy] * phase

            C_k[i, j] = sum_k

    # return to position space
    res = np.fft.ifft2(C_k)
    return res.real


# parameters setup
N = 64
L = 5.0
theta = 1.5
lam = 1.0

x = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y = np.meshgrid(x, x)

# frequencies grid
k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
KX, KY = np.meshgrid(k, k)

# initial gaussian field
phi = np.exp(-(X**2 + Y**2))

# classical phi^3 interaction
phi_classical = (lam / 6.0) * (phi**3)

# non-commutative phi star phi star phi
phi2 = moyal_star(phi, phi, KX, KY, theta)
phi3 = moyal_star(phi2, phi, KX, KY, theta)
phi_noncomm = (lam / 6.0) * phi3

# plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, phi_classical, levels=30)
plt.title(r"Classical $\phi^3$")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(X, Y, phi_noncomm, levels=30)
plt.title(r"Moyal $\phi \star \phi \star \phi$ (theta = " + str(theta) + ")")
plt.colorbar()

plt.tight_layout()
plt.show()