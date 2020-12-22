import taichi as ti
import numpy as np
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

        self.frame = 0
        self.frame_dt = frame_dt
        self.mpm = mpm

    def create_snowflakes(
        self,
        core_radius=0.03,
        num_rods=8,
        rod_length=0.01,
        rod_thickness_deg=15,
    ):
        mpm = self.mpm
        for i in range(1):

            @ti.func
            def sdf(x):
                phi = ti.atan2(x[1], x[0])
                r = x.norm()

                phi = phi % (2 * math.pi / num_rods)

                x = ti.Vector([r * ti.cos(phi), r * ti.sin(phi)])
                return (r < core_radius + rod_length and phi <
                        math.radians(rod_thickness_deg)) or r < core_radius
                # return r < 0.1

            # mpm.add_texture_2d(initial_offset + 0.25 * (i % 3), 0.2 + 0.15 * i,
            #                    pattern)
            mpm.add_particles_inside_sdf(
                self.initial_offset + self.shaker_width / 2, 0.2 + 0.15 * i,
                sdf, self.res)

    def create_bricks(self):
        mpm = self.mpm
        for i in range(3):
            offset_y = 0.33 * i + self.block_height * 1.2
            mpm.add_cube(
                lower_corner=[self.initial_offset, self.ground_y + offset_y],
                cube_size=[self.shaker_width, self.block_height],
                velocity=[0, 0],
                material=MPMSolver.material_elastic)

            num_tooth = 4
            tooth_width = self.shaker_width / num_tooth / 2
            for k in range(2):
                for j in range(num_tooth - k):
                    offset_x = self.initial_offset + tooth_width * (
                        j * 2 + 0.5) + tooth_width * k * 1.1
                    Y = offset_y + (
                        k - 1
                    ) * self.block_height * 2 + self.block_height + self.ground_y
                    real_tooth = tooth_width / 1.3
                    mpm.add_cube(lower_corner=[offset_x, Y],
                                 cube_size=[real_tooth, self.block_height],
                                 velocity=[0, 0],
                                 material=MPMSolver.material_elastic)

                    mpm.add_ellipsoid(center=[
                        offset_x + real_tooth / 2, Y + k * self.block_height
                    ],
                                      radius=real_tooth / 2 * 1.1,
                                      material=MPMSolver.material_elastic)

    def advance(self):
        self.mpm.step(self.frame_dt)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                          dtype=np.uint32)
        particles = self.mpm.particle_info()
        gui = self.gui
        gui.circles(particles['position'] / [[1, 1]],
                    radius=1,
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

    def run(self, frames):
        print('num_particles:', self.mpm.n_particles[None])
        for i in range(frames):
            print(f"Simulating frame {i} / {frames}")
            self.advance()


sim = ShaderSimulation(frame_dt=1 / 160)

# sim.create_bricks()
sim.create_snowflakes()

sim.run(100)
