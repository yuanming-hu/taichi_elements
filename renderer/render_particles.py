import taichi as ti
import os
import sys
import time

ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8)

from renderer import res, Renderer

renderer = Renderer(dx=1 / 512, shutter_time=0)

with_gui = True
if with_gui:
    gui = ti.GUI('Particle Renderer', res)

spp = 200


def main():
    t = time.time()
    renderer.initialize_particles(
        '/home/yuanming/repos/taichi_elements/demo/sim_2020-08-09_19-50-10/00010.npz'
    )
    print('Average particle_list_length',
          renderer.average_particle_list_length())
    img = renderer.render_frame(spp=spp)
    while gui:
        gui.set_image(img)
        gui.show()

    print(f'Frame rendered. {spp} take {time.time() - t} s.')


if __name__ == '__main__':
    main()
