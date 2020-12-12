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

E_scale = 200
dt_scale = 1 / E_scale ** 0.5
mpm = MPMSolver(res=(res, res), E_scale=E_scale, dt_scale=dt_scale, unbounded=True)

pattern = 1 - ti.imread('snowflake.png')[:, :, 1] * (1 / 255.0)
dsize = 128
pattern = cv2.resize(pattern, dsize=(dsize, dsize), interpolation=cv2.INTER_CUBIC)

space = 0.7

for i in range(0):
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

    mpm.add_ellipsoid(center=[0.175 + offset_x, 0.3], radius=0.07, material=MPMSolver.material_elastic)
    
    
    # mpm.add_cube(lower_corner=[0.275 + offset_x, 0.1],
    #              cube_size=[0.01, 0.4],
    #              velocity=[0, 0],
    #              material=MPMSolver.material_elastic)
        
    mpm.add_ellipsoid(center=[0.275 + offset_x, 0.5], radius=0.05, material=MPMSolver.material_elastic)

omega = 27
mag = 0.02
initial_offset = 0.1
shaker_width = 0.8
ground_y = 0.2

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
        '''
        if p[0] < 0.
            omega = 2 * omega_step
            mpm.grid_v
            mpm.grid_v[I] = ti.Vector([omega * 0.01 * ti.sin(t * omega), 0.0])
       '''
        if p[0] < pos_left:
            mpm.grid_v[I][0] = shaker_v
        if p[0] > pos_right:
            mpm.grid_v[I][0] = shaker_v

mpm.grid_postprocess.append(vibrate)

for i in range(10):
    mpm.add_texture(initial_offset + 0.25 * (i % 3), 0.2 + 0.15 * i, pattern)

print(mpm.n_particles[None])

frame_dt = 1 / 60

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
    gui.show(f'outputs4/{frame:06d}.png' if write_to_disk else None)
