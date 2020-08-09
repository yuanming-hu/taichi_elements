import taichi as ti
import os
import sys
import time

ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8)

from renderer import initialize, render_frame, res

with_gui = True
if with_gui:
    gui = ti.GUI('Particle Renderer', res)
    
spp = 200


def output_fn(f):
    return f'rendered/{f:05d}.png'

def main():
    for f in range(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])):
        print(f'frame {f}')
        t = time.time()
        fn = output_fn(f)
        if os.path.exists(output_fn(f)):
            continue
        initialize(f=f)
        img = render_frame(f, spp=spp)
        ti.imwrite(img, fn)
        if gui:
            gui.set_image(img)
            gui.show()

        print(f'Frame rendered. {spp} take {time.time() - t} s.')


if __name__ == '__main__':
    main()
