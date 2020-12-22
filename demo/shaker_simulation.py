import taichi as ti
import numpy as np
import cv2
import utils
import math
from engine.mpm_solver import MPMSolver


class ShaderSimulation:
    def __init__(self,
                 res=256,
                 gui_scale=3,
                 frame_dt=1 / 160,
                 shaker_omega=200,
                 shaker_magnitude=0.005):
        ti.init(arch=ti.cuda)  # Try to run on GPU
        self.write_to_disk = True

        self.res = res
        self.gui_scale = gui_scale

        self.gui = ti.GUI("Taichi Elements",
                          res=(self.res * self.gui_scale,
                               self.res * self.gui_scale),
                          background_color=0x112F41)

        self.E_scale = 20
        dt_scale = 1 / self.E_scale**0.5
        mpm = MPMSolver(res=(self.res, self.res),
                        E_scale=self.E_scale,
                        dt_scale=dt_scale,
                        unbounded=True)
        mpm.set_gravity([0, -1])

        self.omega = shaker_omega
        self.mag = shaker_magnitude
        initial_offset = 0.1
        self.initial_offset = initial_offset
        shaker_width = 0.3
        self.shaker_width = shaker_width
        ground_y = 0.1
        self.ground_y = ground_y
        block_height = 0.1
        self.block_height = block_height

        for i in range(3):
            offset_y = 0.33 * i + block_height * 1.2
            mpm.add_cube(
                lower_corner=[self.initial_offset, ground_y + offset_y],
                cube_size=[shaker_width, block_height],
                velocity=[0, 0],
                material=MPMSolver.material_elastic)

            num_tooth = 4
            tooth_width = shaker_width / num_tooth / 2
            for k in range(2):
                for j in range(num_tooth - k):
                    offset_x = initial_offset + tooth_width * (
                        j * 2 + 0.5) + tooth_width * k * 1.1
                    Y = offset_y + (
                        k - 1) * block_height * 2 + block_height + ground_y
                    real_tooth = tooth_width / 1.3
                    mpm.add_cube(lower_corner=[offset_x, Y],
                                 cube_size=[real_tooth, block_height],
                                 velocity=[0, 0],
                                 material=MPMSolver.material_elastic)

                    mpm.add_ellipsoid(center=[
                        offset_x + real_tooth / 2, Y + k * block_height
                    ],
                                      radius=real_tooth / 2 * 1.1,
                                      material=MPMSolver.material_elastic)

        @ti.kernel
        def vibrate(t: ti.f32, dt: ti.f32):
            for I in ti.grouped(mpm.grid_m):
                p = I * mpm.dx
                if p[1] < ground_y:
                    mpm.grid_v[I].y = 0
                slab_offset = ti.sin(
                    t * self.omega) * self.mag + initial_offset
                pos_left = slab_offset
                pos_right = pos_left + shaker_width
                shaker_v = self.omega * ti.cos(t * self.omega) * self.mag
                if p[0] < pos_left:
                    mpm.grid_v[I][0] = shaker_v
                if p[0] > pos_right:
                    mpm.grid_v[I][0] = shaker_v

        mpm.grid_postprocess.append(vibrate)

        print('num_particles:', mpm.n_particles[None])

        self.frame = 0
        self.frame_dt = frame_dt
        self.mpm = mpm

    def advance(self):
        self.mpm.step(self.frame_dt)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                          dtype=np.uint32)
        particles = self.mpm.particle_info()
        gui = self.gui
        gui.circles(particles['position'] / [[1, 1]],
                    radius=2,
                    color=colors[particles['material']])
        gui.line(begin=(0, self.ground_y),
                 end=(1, self.ground_y),
                 radius=3,
                 color=0xFFFFFF)
        offset = math.sin((self.frame + 0.5) * self.frame_dt *
                          self.omega) * self.mag + self.initial_offset
        gui.line(begin=(offset, self.ground_y),
                 end=(offset, 1.0),
                 radius=3,
                 color=0xFFFFFF)
        gui.line(begin=(offset + self.shaker_width, self.ground_y),
                 end=(offset + self.shaker_width, 1.0),
                 radius=3,
                 color=0xFFFFFF)
        gui.show(
            f'outputs5/{self.frame:06d}.png' if self.write_to_disk else None)

        self.frame += 1


shaker = ShaderSimulation(frame_dt=1 / 160)

for i in range(100):
    shaker.advance()
