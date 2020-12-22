import utils
from shaker_simulation import ShakerSimulation

folder = utils.create_output_folder(prefix='shaker')

sim = ShakerSimulation(output_folder=folder, frame_dt=1 / 160, shaker_width=0.7)

# sim.create_bricks()
sim.create_snowflakes(num_snowflakes=10,
                      core_radius=0.04,
                      num_rods=7,
                      rod_length=0.03,
                      rod_thickness=0.015,
                      end_radius=0.015)

sim.run(frames=20)

from IPython.display import Video
Video(f"{folder}/video.mp4")
