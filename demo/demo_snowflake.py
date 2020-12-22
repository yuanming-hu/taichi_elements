import taichi as ti
import numpy as np
import cv2
import utils
import math
from engine.mpm_solver import MPMSolver

write_to_disk = True

ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=1024, background_color=0x112F41)

E_scale = 200
dt_scale = 3 / E_scale ** 0.5
mpm = MPMSolver(res=(256, 256), E_scale=E_scale, dt_scale=dt_scale, unbounded=True)

pattern = 1 - ti.imread('snowflake.png')[:, :, 1] * (1 / 255.0)
pattern = cv2.resize(pattern, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

mpm.add_surface_collider(point=(0, -0.6),
                         normal=(0.0, 1),
                         surface=mpm.surface_slip)

mpm.add_surface_collider(point=(1.4, 0),
                         normal=(-1, 0),
                         surface=mpm.surface_slip)

mpm.add_surface_collider(point=(-0.6, 0),
                         normal=(1, 0),
                         surface=mpm.surface_slip)

for i in range(5):
    mpm.add_texture_2d(-0.1 + 0.10 * i % 3, -0.1 + 0.6 * i, pattern)

print(mpm.n_particles[None])

for frame in range(500):
    mpm.step(8e-3)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'] * 0.5 + 0.3,
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'outputs/{frame:06d}.png' if write_to_disk else None)
