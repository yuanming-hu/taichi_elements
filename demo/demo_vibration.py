import taichi as ti
import numpy as np
import cv2
import utils
import math
from engine.mpm_solver import MPMSolver

write_to_disk = True

ti.init(arch=ti.cuda)  # Try to run on GPU

res = 128

gui = ti.GUI("Taichi Elements", res=(res * 6 * 2, res * 3 * 2), background_color=0x112F41)

E_scale = 50
dt_scale = 1 / E_scale ** 0.5
mpm = MPMSolver(res=(res, res), E_scale=E_scale, dt_scale=dt_scale, unbounded=True)

for i in range(6):
    mpm.add_cube(lower_corner=[0.15 + 0.3 * i, 0.1],
                 cube_size=[0.05, 0.8],
                 velocity=[0, 0],
                 material=MPMSolver.material_elastic)

omega_step = 3

@ti.kernel
def vibrate(t: ti.f32, dt: ti.f32):
    for I in ti.grouped(mpm.grid_m):
        p = I * mpm.dx
        if p[1] < 0.2:
            omega = (p[0] // 0.3 + 1) * omega_step
            mpm.grid_v[I] = ti.Vector([omega * 0.01 * ti.sin(t * omega), 0.0])

mpm.grid_postprocess.append(vibrate)

print(mpm.n_particles[None])

for frame in range(500):
    mpm.step(1 / 60)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'] / [[2, 1]],
                radius=2,
                color=colors[particles['material']])
    gui.line(begin=(0, 0.2), end=(1, 0.2), radius=4, color=0xFFFFFF)
    for i in range(6):
        gui.text(f'omega={(i + 1) * omega_step:.2f}/sec', (0.15 * i + 0.05, 0.1))
    gui.show(f'outputs/{frame:06d}.png' if write_to_disk else None)
