import taichi as ti
import numpy as np
import cv2
import utils
import math
from engine.mpm_solver import MPMSolver

write_to_disk = True

ti.init(arch=ti.cuda)  # Try to run on GPU

res = 128

gui = ti.GUI("Taichi Elements",
             res=(res * 15, res * 6),
             background_color=0x112F41)

E_scale = 200
dt_scale = 1 / E_scale**0.5
mpm = MPMSolver(res=(res, res),
                E_scale=E_scale,
                dt_scale=dt_scale,
                unbounded=True)

space = 0.7

for i in range(3):
    offset_x = space * i - 0.1
    mpm.add_cube(lower_corner=[0.15 + offset_x, 0.35],
                 cube_size=[0.05, 0.5],
                 velocity=[0, 0],
                 material=MPMSolver.material_elastic)

    mpm.add_cube(lower_corner=[0.15 + offset_x + 0.2, 0.25],
                 cube_size=[0.1, 0.6],
                 velocity=[0, 0],
                 material=MPMSolver.material_elastic)

    mpm.add_cube(lower_corner=[0.18 + offset_x + 0.1, 0.25],
                 cube_size=[0.07, 0.05],
                 velocity=[0, 0],
                 material=MPMSolver.material_elastic)

    mpm.add_ellipsoid(center=[0.175 + offset_x, 0.3],
                      radius=0.07,
                      material=MPMSolver.material_elastic)

    # mpm.add_cube(lower_corner=[0.275 + offset_x, 0.1],
    #              cube_size=[0.01, 0.4],
    #              velocity=[0, 0],
    #              material=MPMSolver.material_elastic)

    mpm.add_ellipsoid(center=[0.275 + offset_x, 0.5],
                      radius=0.05,
                      material=MPMSolver.material_elastic)

omega_step = 27


@ti.kernel
def vibrate(t: ti.f32, dt: ti.f32):
    for I in ti.grouped(mpm.grid_m):
        p = I * mpm.dx
        if p[1] > 0.8:
            omega = (p[0] // space + 1) * omega_step
            mpm.grid_v[I] = ti.Vector([omega * 0.01 * ti.sin(t * omega), 0.0])


mpm.grid_postprocess.append(vibrate)

print(mpm.n_particles[None])

for frame in range(500):
    mpm.step(1 / 60)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'] / [[2.5, 1]],
                radius=2,
                color=colors[particles['material']])
    gui.line(begin=(0, 0.8), end=(1, 0.8), radius=4, color=0xFFFFFF)
    for i in range(3):
        gui.text(f'omega={(i + 1) * omega_step:.2f}/sec',
                 (space / 2.5 * i + 0.05, 0.1))
    gui.show(f'outputs3/{frame:06d}.png' if write_to_disk else None)
