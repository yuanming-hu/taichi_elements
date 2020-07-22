import taichi as ti
import math
import time
import numpy as np
from plyfile import PlyData, PlyElement
import os
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = False

# Try to run on GPU
ti.init(arch=ti.cuda, kernel_profiler=True, use_unified_memory=False, device_memory_GB=8)

gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)

output_dir = 'output_particles'
os.makedirs(output_dir, exist_ok=True)


def load_mesh(fn, scale, offset):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale + offset[2]

    print('loaded')

    return triangles


R = 512

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=True)

mpm.add_surface_collider(point=(0, 0, 0), normal=(0, 1, 0), surface=mpm.surface_slip, friction=1.5)

triangles = load_mesh('taichi.ply', scale=0.02, offset=(0.5, 0.6, 0.5))
mpm.add_mesh(triangles=triangles,
             material=MPMSolver.material_elastic,
             color=0xDDEEFF,
             velocity=(0, -1, 0))

triangles = load_mesh('nvidia.ply', scale=0.02, offset=(0.5, 0.3, 0.5))
mpm.add_mesh(triangles=triangles,
             material=MPMSolver.material_elastic,
             color=0x76B90B,
             velocity=(0, -1, 0))


mpm.set_gravity((0, -25, 0))

n_balls = 5
water_ball_start = 100

snow_brick_start = 150


def add_water_ball(i):
    circle_radius = 0.3
    v = 2
    ang = i * math.pi * 2 / n_balls
    mpm.add_ellipsoid(center=(0.5 + circle_radius * math.cos(ang), 0.5,
                              0.5 + circle_radius * math.sin(ang)),
                      radius=0.03,
                      material=mpm.material_water,
                      velocity=(-v * math.cos(ang), 1.4, -v * math.sin(ang)),
                      color=0x88DDFF)
    

def add_snow_brick(i):
    f = i
    mpm.add_cube(lower_corner=(0.3 + f * 0.05, 0.5 + f * 0.1, 0.3), cube_size=(0.1, 0.03, 0.05),
                      material=mpm.material_snow,
                      velocity=(0, -1, 0),
                      color=0xFFFFFF)


def visualize(particles):
    np_x = particles['position'] / 1.0
    
    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])
    
    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    
    gui.circles(screen_pos, radius=0.8, color=particles['color'])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)

for frame in range(1500):
    if water_ball_start <= frame < water_ball_start + n_balls:
        add_water_ball(frame - water_ball_start)

    if snow_brick_start <= frame < snow_brick_start + n_balls:
        add_snow_brick(frame - snow_brick_start)
        
    if frame == 50:
        triangles = load_mesh('mlsmpm.ply', scale=0.05, offset=(0.5, 0.5, 0.8))
        mpm.add_mesh(triangles=triangles,
                     material=MPMSolver.material_sand,
                     color=0xFF11CC,
                     velocity=(0, -1, 0))

    print(f'frame {frame}')
    mpm.step(4e-3, print_stat=True)
    if frame % 1 == 0:
        particles = mpm.particle_info()
        visualize(particles)

    if write_to_disk:
        particles = mpm.particle_info()
        output_fn = f'{output_dir}/{frame:05d}.npz'
        np.savez_compressed(output_fn,
                            x=particles['position'],
                            v=particles['velocity'],
                            c=particles['color'])
