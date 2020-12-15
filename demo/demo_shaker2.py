import taichi as ti
import numpy as np
import cv2
import utils
import math
from engine.mpm_solver import MPMSolver

write_to_disk = True

ti.init(arch=ti.cuda)  # Try to run on GPU

res = 256
gui_scale = 3

gui = ti.GUI("Taichi Elements", res=(res * gui_scale, res * gui_scale), background_color=0x112F41)

E_scale = 20
dt_scale = 1 / E_scale ** 0.5
mpm = MPMSolver(res=(res, res), E_scale=E_scale, dt_scale=dt_scale, unbounded=True)

pattern = 1 - ti.imread('snowflake.png')[:, :, 1] * (1 / 255.0)
dsize = 128
pattern = cv2.resize(pattern, dsize=(dsize, dsize), interpolation=cv2.INTER_CUBIC)

space = 0.7

omega = 200
mag = 0.005
initial_offset = 0.1
shaker_width = 0.3
ground_y = 0.2
block_height = 0.1

for i in range(3):
    offset_y = 0.33 * i + block_height * 1.2
    mpm.add_cube(lower_corner=[initial_offset, ground_y + offset_y],
                 cube_size=[shaker_width, block_height],
                 velocity=[0, 0],
                 material=MPMSolver.material_elastic)
    
    num_tooth = 4
    tooth_width = shaker_width / num_tooth / 2
    for k in range(2):
        for j in range(num_tooth - k):
            offset_x = initial_offset + tooth_width * (j * 2 + 0.5) + tooth_width * k * 1.1
            Y = offset_y + (k - 1) * block_height * 2 + block_height + ground_y
            real_tooth = tooth_width / 1.3
            mpm.add_cube(lower_corner=[offset_x,  Y],
                         cube_size=[real_tooth, block_height],
                         velocity=[0, 0],
                         material=MPMSolver.material_elastic)

            mpm.add_ellipsoid(center=[offset_x + real_tooth / 2, Y + k * block_height], radius=real_tooth / 2 * 1.1, material=MPMSolver.material_elastic)

@ti.kernel
def vibrate(t: ti.f32, dt: ti.f32):
    for I in ti.grouped(mpm.grid_m):
        p = I * mpm.dx
        if p[1] < 0.2:
            mpm.grid_v[I].y = 0
        slab_offset = ti.sin(t * omega) * mag + initial_offset
        pos_left = slab_offset
        pos_right = pos_left + shaker_width
        shaker_v = omega * ti.cos(t * omega) * mag
        if p[0] < pos_left:
            mpm.grid_v[I][0] = shaker_v
        if p[0] > pos_right:
            mpm.grid_v[I][0] = shaker_v

mpm.grid_postprocess.append(vibrate)

print(mpm.n_particles[None])

frame_dt = 1 / 160

for frame in range(500):
    mpm.step(frame_dt)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'] / [[1, 1]],
                radius=2,
                color=colors[particles['material']])
    gui.line(begin=(0, 0.2), end=(1, 0.2), radius=3, color=0xFFFFFF)
    offset = math.sin((frame + 0.5) * frame_dt * omega) * mag + initial_offset
    gui.line(begin=(offset, 0.2), end=(offset, 1.0), radius=3, color=0xFFFFFF)
    gui.line(begin=(offset + shaker_width, ground_y), end=(offset + shaker_width, 1.0), radius=3, color=0xFFFFFF)
    gui.show(f'outputs5/{frame:06d}.png' if write_to_disk else None)
