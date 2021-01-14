import taichi as ti
import numpy as np
import math
import time
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi

res = 1280, 720
aspect_ratio = res[0] / res[1]

max_ray_depth = 4
use_directional_light = True

fov = 0.23
dist_limit = 100
# TODO: why doesn't it render normally when shutter_begin = -1?
shutter_begin = -0.5

exposure = 1.5
light_direction = [1.2, 0.3, 0.7]
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]


@ti.data_oriented
class Renderer:
    def __init__(self,
                 dx=1 / 1024,
                 sphere_radius=0.3 / 1024,
                 render_voxel=False,
                 shutter_time=1e-3,
                 taichi_logo=True):
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.taichi_logo = taichi_logo

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.voxel_grid_density = ti.field(dtype=ti.f32)
        self.voxel_has_particle = ti.field(dtype=ti.i32)

        self.particle_x = ti.Vector.field(3, dtype=ti.f32)
        self.particle_v = ti.Vector.field(3, dtype=ti.f32)
        self.particle_color = ti.Vector.field(3, dtype=ti.u8)
        self.pid = ti.field(ti.i32)
        self.num_particles = ti.field(ti.i32, shape=())

        self.render_voxel = render_voxel

        self.voxel_edges = 0.1

        self.particle_grid_res = 2048

        self.dx = dx
        self.inv_dx = 1 / self.dx

        self.camera_pos = ti.Vector([float(0.0), float(1.0), float(5.0)])

        self.supporter = 2
        self.shutter_time = shutter_time  # usually half the frame time
        self.sphere_radius = sphere_radius
        self.particle_grid_offset = [
            -self.particle_grid_res // 2 for _ in range(3)
        ]

        self.voxel_grid_res = self.particle_grid_res
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]
        self.max_num_particles_per_cell = 8192 * 1024
        self.max_num_particles = 1024 * 1024 * 256

        self.voxel_dx = self.dx
        self.voxel_inv_dx = 1 / self.voxel_dx

        assert self.sphere_radius * 2 < self.dx

        ti.root.dense(ti.ij, res).place(self.color_buffer)

        self.particle_bucket = ti.root.pointer(ti.ijk,
                                               self.particle_grid_res // 8)
        self.particle_bucket.dense(ti.ijk, 8).dynamic(
            ti.l, self.max_num_particles_per_cell,
            chunk_size=64).place(self.pid,
                                 offset=self.particle_grid_offset + [0])

        ti.root.pointer(ti.ijk, self.particle_grid_res // 8).dense(
            ti.ijk, 8).place(self.voxel_has_particle,
                             offset=self.particle_grid_offset)
        voxel_block = ti.root.pointer(ti.ijk, self.voxel_grid_res // 8)
        voxel_block.dense(ti.ijk, 8).place(self.voxel_grid_density,
                                           offset=voxel_grid_offset)

        ti.root.dense(ti.l,
                      self.max_num_particles).place(self.particle_x,
                                                    self.particle_v,
                                                    self.particle_color)

    @ti.func
    def inside_grid(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2

    # The dda algorithm requires the voxel grid to have one surrounding layer of void region
    # to correctly render the outmost voxel faces
    @ti.func
    def inside_grid_loose(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 - 1 and ipos.max(
        ) <= self.voxel_grid_res // 2

    @ti.func
    def query_density(self, ipos):
        inside = self.inside_grid(ipos)
        ret = 0.0
        if inside:
            ret = self.voxel_grid_density[ipos]
        else:
            ret = 0.0
        return ret

    @ti.func
    def voxel_color(self, pos):
        p = pos * self.inv_dx

        p -= ti.floor(p)

        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if p[i] < boundary or p[i] > 1 - boundary:
                count += 1
        f = 0.0
        if count >= 2:
            f = 1.0
        return ti.Vector([0.9, 0.8, 1.0]) * (1.3 - 1.2 * f)

    @ti.func
    def sdf(self, o):
        dist = 0.0
        if ti.static(self.supporter == 0):
            o -= ti.Vector([0.5, 0.002, 0.5])
            p = o
            h = 0.02
            ra = 0.29
            rb = 0.005
            d = (ti.Vector([p[0], p[2]]).norm() - 2.0 * ra + rb, abs(p[1]) - h)
            dist = min(max(d[0], d[1]), 0.0) + ti.Vector(
                [max(d[0], 0.0), max(d[1], 0)]).norm() - rb
        elif ti.static(self.supporter == 1):
            o -= ti.Vector([0.5, 0.002, 0.5])
            dist = (o.abs() - ti.Vector([0.5, 0.02, 0.5])).max()
        else:
            dist = o[1] + 0.002

        return dist

    @ti.func
    def ray_march(self, p, d):
        j = 0
        dist = 0.0
        limit = 200
        while j < limit and self.sdf(p +
                                     dist * d) > 1e-8 and dist < dist_limit:
            dist += self.sdf(p + dist * d)
            j += 1
        if dist > dist_limit:
            dist = inf
        return dist

    @ti.func
    def sdf_normal(self, p):
        d = 1e-3
        n = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self.sdf(inc) - self.sdf(dec))
        return n.normalized()

    @ti.func
    def sdf_color(self, p):
        scale = 0.0
        if ti.static(self.taichi_logo):
            scale = 0.4
            if inside_taichi(ti.Vector([p[0], p[2]])):
                scale = 1
        else:
            scale = 1.0
        return ti.Vector([0.3, 0.5, 0.7]) * scale

    # Digital differential analyzer for the grid visualization (render_voxels=True)
    @ti.func
    def dda_voxel(self, eye_pos, d):
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6
        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                                 d)
        hit_distance = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        if inter:
            near = max(0, near)

            pos = eye_pos + d * (near + 5 * eps)

            o = self.voxel_inv_dx * pos
            ipos = int(ti.floor(o))
            dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
            running = 1
            i = 0
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            while running:
                last_sample = int(self.query_density(ipos))
                if not self.inside_particle_grid(ipos):
                    running = 0

                if last_sample:
                    mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) -
                            rsign * 0.5) * rinv
                    hit_distance = mini.max() * self.voxel_dx + near
                    hit_pos = eye_pos + hit_distance * d
                    c = self.voxel_color(hit_pos)
                    running = 0
                else:
                    mm = ti.Vector([0, 0, 0])
                    if dis[0] <= dis[1] and dis[0] < dis[2]:
                        mm[0] = 1
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        mm[1] = 1
                    else:
                        mm[2] = 1
                    dis += mm * rsign * rinv
                    ipos += mm * rsign
                    normal = -mm * rsign
                i += 1
        return hit_distance, normal, c

    @ti.func
    def inside_particle_grid(self, ipos):
        pos = ipos * self.dx
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]

    # DDA for the particle visualization (render_voxels=False)
    @ti.func
    def dda_particle(self, eye_pos, d, t):
        # bounding box
        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]

        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6

        inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                                 d)
        near = max(0, near)

        closest_intersection = inf

        if inter:
            pos = eye_pos + d * (near + eps)

            rinv = 1.0 / d
            rsign = ti.Vector([0, 0, 0])
            for i in ti.static(range(3)):
                if d[i] > 0:
                    rsign[i] = 1
                else:
                    rsign[i] = -1

            o = self.inv_dx * pos
            ipos = ti.floor(o).cast(int)
            dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
            running = 1
            # DDA for voxels with at least one particle
            while running:
                inside = self.inside_particle_grid(ipos)

                if inside:
                    # once we actually intersect with a voxel that contains at least one particle, loop over the particle list
                    num_particles = self.voxel_has_particle[ipos]
                    if num_particles != 0:
                        num_particles = ti.length(
                            self.pid.parent(),
                            ipos - ti.Vector(self.particle_grid_offset))
                    for k in range(num_particles):
                        p = self.pid[ipos, k]
                        v = self.particle_v[p]
                        x = self.particle_x[p] + t * v
                        color = ti.cast(self.particle_color[p],
                                        ti.u32) * (1 / 255.0)
                        # ray-sphere intersection
                        dist, poss = intersect_sphere(eye_pos, d, x,
                                                      self.sphere_radius)
                        hit_pos = poss
                        if dist < closest_intersection and dist > 0:
                            hit_pos = eye_pos + dist * d
                            closest_intersection = dist
                            normal = (hit_pos - x).normalized()
                            c = color
                else:
                    running = 0
                    normal = [0, 0, 0]

                if closest_intersection < inf:
                    running = 0
                else:
                    # hits nothing. Continue ray marching
                    mm = ti.Vector([0, 0, 0])
                    if dis[0] <= dis[1] and dis[0] <= dis[2]:
                        mm[0] = 1
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        mm[1] = 1
                    else:
                        mm[2] = 1
                    dis += mm * rsign * rinv
                    ipos += mm * rsign

        return closest_intersection, normal, c

    @ti.func
    def next_hit(self, pos, d, t):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        if ti.static(self.render_voxel):
            closest, normal, c = self.dda_voxel(pos, d)
        else:
            closest, normal, c = self.dda_particle(pos, d, t)

        if d[2] != 0:
            ray_closest = -(pos[2] + 5.5) / d[2]
            if ray_closest > 0 and ray_closest < closest:
                closest = ray_closest
                normal = ti.Vector([0.0, 0.0, 1.0])
                c = ti.Vector([0.6, 0.7, 0.7])

        ray_march_dist = self.ray_march(pos, d)
        if ray_march_dist < dist_limit and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest)
            c = self.sdf_color(pos + d * closest)

        return closest, normal, c

    @ti.kernel
    def render(self):
        ti.block_dim(256)
        for u, v in self.color_buffer:
            pos = self.camera_pos
            d = ti.Vector([
                (2 * fov * (u + ti.random(ti.f32)) / res[1] -
                 fov * aspect_ratio - 1e-5),
                2 * fov * (v + ti.random(ti.f32)) / res[1] - fov - 1e-5, -1.0
            ])
            d = d.normalized()
            t = (ti.random() + shutter_begin) * self.shutter_time

            contrib = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_sky = 1
            ray_depth = 0

            while depth < max_ray_depth:
                closest, normal, c = self.next_hit(pos, d, t)
                hit_pos = pos + closest * d
                depth += 1
                ray_depth = depth
                if normal.norm() != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c

                    if ti.static(use_directional_light):
                        dir_noise = ti.Vector([
                            ti.random() - 0.5,
                            ti.random() - 0.5,
                            ti.random() - 0.5
                        ]) * light_direction_noise
                        direct = (ti.Vector(light_direction) +
                                  dir_noise).normalized()
                        dot = direct.dot(normal)
                        if dot > 0:
                            dist, _, _ = self.next_hit(pos, direct, t)
                            if dist > dist_limit:
                                contrib += throughput * ti.Vector(
                                    light_color) * dot
                else:  # hit sky
                    hit_sky = 1
                    depth = max_ray_depth

                max_c = throughput.max()
                if ti.random() > max_c:
                    depth = max_ray_depth
                    throughput = [0, 0, 0]
                else:
                    throughput /= max_c

            if hit_sky:
                if ray_depth != 1:
                    # contrib *= max(d[1], 0.05)
                    pass
                else:
                    # directly hit sky
                    pass
            else:
                throughput *= 0

            # contrib += throughput
            self.color_buffer[u, v] += contrib

    @ti.kernel
    def initialize_particle_grid(self):
        for p in range(self.num_particles[None]):
            x = self.particle_x[p]
            ipos = ti.floor(x * self.inv_dx).cast(ti.i32)
            print(p, x, ipos)
            

            for i in range(-1, 1):
                for j in range(-1, 1):
                    for k in range(-1, 1):
                        offset = ti.Vector([i, j, k])
                        box_ipos = ipos + offset
                        if self.inside_particle_grid(box_ipos):
                            len = ti.length(
                                self.pid.parent(), box_ipos -
                                                   ti.Vector(self.particle_grid_offset))
                            print('before', len)
                            ti.append(
                                self.pid.parent(), box_ipos -
                                ti.Vector(self.particle_grid_offset), p)
                            print('after', len)

    @ti.kernel
    def copy(self, img: ti.ext_arr(), samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / res[0]
            v = 1.0 * j / res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0])**2 +
                (v - self.vignette_center[1])**2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                img[i, j, c] = ti.sqrt(self.color_buffer[i, j][c] * darken *
                                       exposure / samples)

    @ti.kernel
    def initialize_particle(self, x: ti.ext_arr(), v: ti.ext_arr(),
                            color: ti.ext_arr(), begin: ti.i32, end: ti.i32):
        for i in range(begin, end):
            for c in ti.static(range(3)):
                self.particle_x[i][c] = x[i - begin, c]
                self.particle_v[i][c] = v[i - begin, c]
                self.particle_color[i][c] = color[i - begin, c]

    @ti.kernel
    def total_non_empty_voxels(self) -> ti.i32:
        counter = 0

        for I in ti.grouped(self.voxel_has_particle):
            if self.voxel_has_particle[I]:
                counter += 1

        return counter

    @ti.kernel
    def total_inserted_particles(self) -> ti.i32:
        counter = 0

        for I in ti.grouped(self.voxel_has_particle):
            if self.voxel_has_particle[I]:
                num_particles = ti.length(
                    self.pid.parent(),
                    I - ti.Vector(self.particle_grid_offset))
                counter += num_particles

        return counter

    def reset(self):
        self.particle_bucket.deactivate_all()
        self.voxel_grid_density.snode.parent(n=2).deactivate_all()
        self.voxel_has_particle.snode.parent(n=2).deactivate_all()
        self.color_buffer.fill(0)

    def initialize_particles_from_taichi_elements(self, particle_fn):
        self.reset()

        from particle_io import ParticleIO

        np_x, np_v, np_color = ParticleIO.read_particles_3d(particle_fn)
        num_part = len(np_x)

        # TODO: remove this hack
        # np_x = np_x * 0.5

        assert num_part <= self.max_num_particles

        for i in range(3):
            # bbox values must be multiples of self.dx
            # bbox values are the min and max particle coordinates, with 3 self.dx margin
            self.bbox[0][i] = (math.floor(np_x[:, i].min() * self.inv_dx) -
                               3.0) * self.dx
            self.bbox[1][i] = (math.floor(np_x[:, i].max() * self.inv_dx) +
                               3.0) * self.dx
            print(f'Bounding box dim {i}: {self.bbox[0][i]} {self.bbox[1][i]}')

        # TODO: assert bounds

        self.num_particles[None] = num_part
        print('num_input_particles =', num_part)

        slice_size = 1000000
        num_slices = (num_part + slice_size - 1) // slice_size
        for i in range(num_slices):
            begin = slice_size * i
            end = min(num_part, begin + slice_size)
            print(begin, end)
            self.initialize_particle(np_x[begin:end], np_v[begin:end],
                                     np_color[begin:end], begin, end)
        self.initialize_particle_grid()

    def render_frame(self, spp):
        last_t = 0
        for i in range(1, 1 + spp):
            self.render()

            interval = 20
            if i % interval == 0:
                if last_t != 0:
                    ti.sync()
                    print("time per spp = {:.2f} ms".format(
                        (time.time() - last_t) * 1000 / interval))
                last_t = time.time()

        img = np.zeros((res[0], res[1], 3), dtype=np.float32)
        self.copy(img, spp)
        return img
