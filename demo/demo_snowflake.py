import taichi as ti
import numpy as np
import utils
import math
from engine.mpm_solver import MPMSolver

write_to_disk = False

ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(1024, 1024))

pattern = 1 - ti.imread('snowflake.png')[:, :, 1] * (1 / 255.0)
print(pattern.shape)
print(pattern.max())
print(pattern.min())

# for i in range(3):
mpm.add_texture(0.1, 0.1, pattern)

for frame in range(500):
    mpm.step(8e-3)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'],
                radius=1,
                color=colors[particles['material']])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
