import taichi as ti
import os
import sys
import time
from pathlib import Path

ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8)

output_folder = sys.argv[5]
os.makedirs(output_folder, exist_ok=True)

from renderer import res, Renderer

renderer = Renderer(dx=1 / 512, shutter_time=2e-3, taichi_logo=False)

with_gui = False
if with_gui:
    gui = ti.GUI('Particle Renderer', res)

spp = 200


def main():
    for f in range(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])):
        print('frame', f, end='')
        output_fn = f'{output_folder}/{f:05d}.png'
        if os.path.exists(output_fn):
            print('skip.')
            continue
        else:
            print('rendering...')
        Path(output_fn).touch()
        t = time.time()
        renderer.initialize_particles(f'{sys.argv[1]}/{f:05d}.npz')
        print('Average particle_list_length',
              renderer.average_particle_list_length())
        img = renderer.render_frame(spp=spp)

        if with_gui:
            gui.set_image(img)
            gui.show(output_fn)
        else:
            ti.imwrite(img, output_fn)

        print(f'Frame rendered. {spp} take {time.time() - t} s.')


if __name__ == '__main__':
    main()
