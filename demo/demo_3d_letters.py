import taichi as ti
import math
import time
import numpy as np
from plyfile import PlyData, PlyElement
import os
import utils
from utils import create_output_folder
from engine.mpm_solver import MPMSolver

write_to_disk = True

# Try to run on GPU
ti.init(arch=ti.cuda, kernel_profiler=True, use_unified_memory=False, device_memory_GB=15)

gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)

output_dir = create_output_folder('particles')


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

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=True, dt_scale=1)

mpm.add_surface_collider(point=(0, 0, 0), normal=(0, 1, 0), surface=mpm.surface_slip, friction=0.5)

triangles = load_mesh('taichi.ply', scale=0.02, offset=(0.5, 0.6, 0.5))

mpm.set_gravity((0, -25, 0))


def visualize(particles):
    np_x = particles['position'] / 1.0
    
    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])
    
    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    
    gui.circles(screen_pos, radius=0.8, color=particles['color'])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
    
counter = 0

for frame in range(15000):
    if frame % 15 == 0:
        if counter < 50:
            mpm.add_mesh(triangles=triangles,
                         material=MPMSolver.material_elastic,
                         color=0xFFFFFF,
                         velocity=(0, -2, 0))

        counter += 1
    
    print(f'frame {frame}')
    mpm.step(2e-3, print_stat=True)
    if frame % 15 == 0:
        particles = mpm.particle_info()
        visualize(particles)

    if write_to_disk:
        particles = mpm.particle_info()
        output_fn = f'{output_dir}/{frame:05d}.npz'
        np.savez_compressed(output_fn,
                            x=particles['position'],
                            v=particles['velocity'],
                            c=particles['color'])
