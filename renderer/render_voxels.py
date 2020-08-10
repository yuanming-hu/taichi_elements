import taichi as ti
import os
import sys
import time

ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8)

from renderer import Renderer, res

grid_down_sample = 8

renderer = Renderer(dx=grid_down_sample / 512, render_voxel=True)

with_gui = True
if with_gui:
    gui = ti.GUI('Voxel Renderer', res)

spp = 200

output_folder = 'rendered_voxels'
os.makedirs(output_folder, exist_ok=True)

def output_fn(f):
    return f'{output_folder}/{f:05d}.png'


@ti.kernel
def load_vdb():
    vdb_id = 0
    a = 0
    ti.runtime_func_call('runtime_vdb_num_leaf_nodes',
                         args=[vdb_id],
                         outputs=[a])

    lower = -renderer.voxel_grid_res // 2
    upper = renderer.voxel_grid_res // 2

    for i in ti.static(range(3)):
        renderer.bbox[0][i] = 10000000
        renderer.bbox[1][i] = -10000000

    for i in range(a):
        x = 0
        y = 0
        z = 0
        ti.runtime_func_call("runtime_vdb_get_coord",
                             args=[vdb_id, i],
                             outputs=[x, y, z])
        x = x // 8 * 8
        y = y // 8 * 8
        z = z // 8 * 8
        for j in range(8**3):
            val = 0.0
            ti.runtime_func_call("runtime_vdb_get_val",
                                 args=[vdb_id, i, j],
                                 outputs=[val])
            p, q, r = x + j // 64, y + j // 8 % 8, z + j % 8
            p = p // grid_down_sample
            q = q // grid_down_sample
            r = r // grid_down_sample
            if renderer.inside_grid(ti.Vector([p, q, r])):
                ti.atomic_min(renderer.bbox[0][0], p)
                ti.atomic_min(renderer.bbox[0][1], q)
                ti.atomic_min(renderer.bbox[0][2], r)
                ti.atomic_max(renderer.bbox[1][0], p)
                ti.atomic_max(renderer.bbox[1][1], q)
                ti.atomic_max(renderer.bbox[1][2], r)
                if val > 0:
                    renderer.voxel_grid_density[p, q, r] = 1

    for i in ti.static(range(3)):
        renderer.bbox[0][i] = (renderer.bbox[0][i] - 2) * renderer.dx
        renderer.bbox[1][i] = (renderer.bbox[1][i] + 2) * renderer.dx


def main():
    for f in range(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])):
        fn = f"{sys.argv[1]}/{f:05d}.nvdb"
        ti.get_runtime().materialize()
        ti.get_runtime().prog.load_vdb(0, fn)

        renderer.reset()

        load_vdb()

        print(f'frame {f}')
        t = time.time()
        fn = output_fn(f)
        img = renderer.render_frame(spp=spp)
        ti.imwrite(img, fn)
        if gui:
            while True:
                gui.set_image(img)
                gui.show()

        print(f'Frame rendered. {spp} take {time.time() - t} s.')


if __name__ == '__main__':
    main()
