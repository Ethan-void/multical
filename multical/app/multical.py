from dataclasses import dataclass
from multical.config.arguments import run_with
from multiprocessing import cpu_count
from typing import  Union


from multical.app.boards import Boards
from multical.app.calibrate import Calibrate
from multical.app.evaluate3d import Evaluate3d
from multical.app.intrinsic import Intrinsic
from multical.app.observe import Observe
from multical.app.rectify import Rectify
from multical.app.triangulate import Triangulate
from multical.app.vis import Vis
from multical.app.world import World
from multical.app.worldmulti import Worldmulti


@dataclass
class Multical:
  """multical - multi camera calibration 
  - calibrate: multi-camera calibration
  - evaluate3d: compare reconstructed and measured world points
  - intrinsic: calibrate separate intrinsic parameters
  - observe: manually annotate synchronized target pixels
  - boards: generate/visualize board images, test detections
  - vis: visualize results of a calibration
  - rectify: generate stereo rectification for camera pairs
  - triangulate: reconstruct synchronized pixels in world coordinates
  - world: anchor relative camera poses to measured world coordinates
  - worldmulti: jointly anchor a fixed rig using multiple cameras
  """ 
  command : Union[
    Calibrate, Intrinsic, Boards, Vis, Rectify, World, Worldmulti, Observe,
    Triangulate, Evaluate3d
  ]
   
  def execute(self):
    return self.command.execute()


def cli():
  run_with(Multical)

if __name__ == '__main__':
  cli()
