import taichi as ti
import time
import numpy as np
from plyfile import PlyData, PlyElement
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = True

# Try to run on GPU
ti.init(arch=ti.cuda, kernel_profiler=True)

gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)


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


triangles = load_mesh('nvidia.ply', scale=0.05, offset=(0.5, 0.5, 0.5))

R = 512

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=False)

# triangles = np.fromfile('triangles.npy', dtype=np.float32)
# triangles = np.reshape(triangles, (len(triangles) // 9, 9)) * 0.306 + 0.501

mpm.add_mesh(triangles=triangles, material=MPMSolver.material_elastic, color=0xFFFF00)
print(f'Num particles={mpm.n_particles[None]}')

mpm.set_gravity((0, -20, 0))

for frame in range(1500):
    print(f'frame {frame}')
    t = time.time()
    step = mpm.total_substeps
    mpm.step(4e-3)
    ti.kernel_profiler_print()
    print(f'sub step time {1000 * (time.time() - t) / (mpm.total_substeps - step)}')
    particles = mpm.particle_info()
    np_x = particles['position'] / 1.0

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1.1, color=particles['color'])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
