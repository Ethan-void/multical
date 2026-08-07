# multical 


Multi-camera calibration using one or more calibration patterns. 

![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/image_view.png)
![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/3d_view.png)

[changelog](https://github.com/saulzar/multical/tree/master/example_boards)

## Install

The software here is presented as a library installable from PyPi `pip install multical`, and installs a script of the same name `multical`.


## Running multical application

The script named `multical` is the primary entry point to run the calibration application. It is a wrapper around several subscripts, namely:
```
usage: multical [-h] {calibrate,intrinsic,boards,show} ...
```

For command line parameters, check sub-command help e.g. `multical calibrate --help`

### Input formats


The default method is to have folders named separately for each camera, with images having corresponding filenames. For example:
```
  - cam1
    - image01.jpg
    - image02.jpg
  - cam2
    - image01.jpg
    - image02.jpg
```    

Camera names and image directories can be also specified by manually specifying camera names, and optionally specifying a pattern for the directory structure. 

`multical calibrate --camera_pattern '{camera}/extrinsic' --cameras cam1 cam2 cam3`

By default the current directory is searched, it can be specified with `--image_path`.

Where `{camera}` is replaced by the camera names 

```
  - cam1
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg    
  - cam2
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg    
    -cam3
    ...
```    

A fixed number of images will be chosen for initial intrinsic calibration (increase for more accuracy at expense of time with `--limit_intrinsic`).

### Outputs

The outputs from `multical calibrate` are written to the `--output_path` which by default is the image path unless specified. The name `calibration` (default) is specified by `--name`.

* `calibration.json` - camera summary, intrinsic parameters, relative camera
  poses, camera rig poses, and final bundle-adjustment quality. The `quality`
  object contains `RMS` (final inlier reprojection RMS in pixels), `RMS_all`,
  and the corresponding observation counts.

* `calibration.log`  - log file of the calibration history
* `calibration.detections.pkl`   - cached calibration pattern detections, making repeated calibrations much faster
* `calibration.pkl`    - serialized workspace containing all the details for visualization, resuming calibration etc.

When `multical intrinsic` is run separately, every camera in
`intrinsic.json` also contains a `quality` object. It records OpenCV's
intrinsic `RMS`, mean and maximum per-view RMS, the number of views, images
and corner observations used, and a per-view breakdown. These extra fields
are ignored when that file is later passed to
`multical calibrate --calibration ... --fix_intrinsic`.

### Calibration targets

Calibration targets supported are checkerboards, ChArUco boards, and AprilGrid boards (as used by Kalibr). Targets are configured by a configuration file with `--boards` and examples can be found in the source tree: [example_boards](https://github.com/saulzar/multical/tree/master/example_boards). Conventional checkerboards are best suited to a single board; use uniquely coded ChArUco or AprilGrid faces for multi-face calibration rigs.

It is a good idea to check your expectation against the configuration specified using an image before calibration `multical boards --boards my_board.yaml --detect my_image.jpeg`, 

### Visualization of output

To install the libraries needed for running visualization (qtpy, pyvistaqt principly) install the `interactive` option, `pip install multical[interactive]` - these may be installed separately depending on your preference (for example with conda).

Visualization can be run by :
`multical vis --workspace_file calibration.pkl`



## Library structure


Multical provides a convenient highlevel interface found in `multical.workspace` which contains most typical useage, from finding images, loading images, initial single camera calibration, board pose extraction, pose initialisation, and finally bundle adjustment optimization and data export.

It is also the best documentation for how to use lower-level library features.

## Synthetic calibration datasets

The repository includes an end-to-end six-camera dataset generator for conventional checkerboards, ChArUco, and AprilGrid:

```bash
uv run python tests/generate_calibration_simulation.py \
  --output-dir charuco_l_split \
  --board-type charuco \
  --board-style l \
  --dataset-mode split \
  --intrinsic-frames 20 \
  --extrinsic-frames 50
```

`--dataset-mode combined` puts synchronized intrinsic-targeted and extrinsic frames in one set for joint calibration. `split` writes separate `intrinsic/` and `extrinsic/` trees. Coded targets support `plane`, `l`, and `triangle`; an ordinary checkerboard is limited to `plane` because identical uncoded faces cannot be distinguished. Each generated dataset contains `boards.yaml`, `ground_truth.json`, a ready-to-run `commands.txt`, and paged contact sheets under `overviews/`. Each overview row shows the same frame from C1 through C6, and all generated camera images are included in filename order. Contact sheets use 320x180 thumbnails and ten synchronized frames per page.

After calibration, run the evaluation command included in `commands.txt`. The evaluator compares `calibration.json` with the synthetic ground truth, prints the report, and automatically writes `evaluation.json` beside `ground_truth.json`. Use `--output path/to/report.json` to choose another destination.

### Stereo rectification

Generate OpenCV stereo-rectification parameters and remap tables for one or more camera pairs:

```bash
uv run multical rectify --calibration calibration.json --pairs C1:C2 C3:C4 C5:C6 --image_path images --frame frame_0000.jpg
```

The default `rectification/` output contains `rectification.json`, one compressed `*_maps.npz` file per pair, and (when `--image_path` is supplied) rectified sample images plus a side-by-side epipolar-line preview. Each pair in `rectification.json` includes `E`, `F`, `R1`, `R2`, `P1`, `P2`, and `Q`; the root `quality` object carries the final calibration RMS copied from `calibration.json`. Here `E` and `F` follow the declared left-to-right transform convention. The RMS is the global multi-camera bundle-adjustment reprojection RMS, not a separately recomputed pairwise `stereoCalibrate` RMS. The default `--alpha -1` lets OpenCV choose the scale; use `--alpha 0` to crop to valid pixels or `--alpha 1` to retain the full field of view. Runtime code can load the four arrays from a map file and pass them to `cv2.remap`.

### World-coordinate extrinsics

Multical estimates a camera rig relative to its master camera. To anchor the whole rig to a measured world coordinate system, create a JSON or YAML control-point file:

```yaml
camera: C1
world_units: meters
world_points:
  - [0.0, 0.0, 0.0]
  - [10.97, 0.0, 0.0]
  - [0.0, 23.77, 0.0]
  - [10.97, 23.77, 0.0]
  - [5.485, 0.0, 0.0]
  - [5.485, 23.77, 0.0]
image_points:
  - [u0, v0]
  - [u1, v1]
  - [u2, v2]
  - [u3, v3]
  - [u4, v4]
  - [u5, v5]
```

Each image point must correspond to the world point on the same row. Then run:

```bash
uv run multical world --calibration calibration.json --correspondences world_points.yaml
```

The resulting `world_extrinsics.json` contains `world_to_camera`, `camera_to_world`, and `position_world` for every camera, together with the PnP inlier count and reprojection error. Four pairs are the minimum; 8-20 well-spread measured control points are recommended.

### World 3D reconstruction

Once `world_extrinsics.json` exists, provide synchronized observations of the same target:

```yaml
sequence: tennis-ball-001
frames:
  - frame: frame_0000
    timestamp: 0.000
    observations:
      C1: [641.2, 358.7]
      C2: [522.8, 361.1]
      C3:
        point: [703.4, 349.8]
        confidence: 0.94
  - frame: frame_0001
    timestamp: 0.0083
    observations:
      C1: [643.0, 355.2]
      C2: [525.1, 357.6]
```

Then triangulate all frames in world coordinates:

```bash
uv run multical triangulate --calibration calibration.json --world_extrinsics world_extrinsics.json --observations ball_observations.yaml
```

The default `triangulation.json` output contains one `[X, Y, Z]` world point per successful frame, cameras used/rejected, per-camera reprojection errors, reprojection RMS, and the maximum triangulation ray angle. At least two synchronized observations are required. With three or more cameras, observations inconsistent with `--reprojection_threshold` are rejected. Frames with insufficient observations, negative depth, poor geometry, or a ray angle below `--min_ray_angle_deg` are reported as failed instead of producing an unreliable point.

AprilGrid detection uses OpenCV's AprilTag 36h11 detector and does not require the Linux-only `apriltags2-ethz` package.


## Non-overlapping case (cameras don't overlap at all in their view)

Thank you very much to Tasnim Tabassum Nova, who contributed a hand-eye calibration mode (configured with the option `is_non_overlapping`).

She has kindly provided the code and a dataset with 6 cameras that did not have overlapping views. The dataset for this project is available [here](https://zenodo.org/records/13294455)
 
<img src="https://github.com/user-attachments/assets/2b58c4cb-cdc6-49ba-b1d0-63af123a6473" width="300" height="300">
<img src="https://github.com/user-attachments/assets/9256dff5-e3c8-463b-a43d-588e74d33182" width="300" height="300">

1. A hand-eye calibration method to address the non-overlapping camera scenario.
2. An iterative approach for calculating intrinsic parameters, which proves more robust for large datasets and reduces the need to select suitable images for intrinsic calibration manually.
3. Functionality to reject outlier poses from further calculations.

The complete visualization for the initial guess for camera extrinsic calibration and the final estimation is available [here](https://chart-studio.plotly.com/~Tabassum_Nova/7/#/plot)
![ex_viz1](https://github.com/user-attachments/assets/f9b0bba8-6e36-42f2-be33-d7ca540b2795)




## FAQ

### How do I make a (physical) board pattern?
Here's my workflow for making board images:

```
multical boards --boards example_boards/charuco_16x22.yaml --paper_size A2 --pixels_mm 10 --write my_images 

Using boards:
charuco_16x22 CharucoBoard {type='charuco', aruco_dict='4X4_1000', aruco_offset=0, size=(16, 22), marker_length=0.01875, square_length=0.025, aruco_params={}}
Wrote my_images/charuco_16x22.png
```

Then open up images/charuco_16x22.png in gimp and print-to-file (pdf) with the margins set to zero and the paper size set to A2. Print pdf to printer or send to print shop.

### Can multical calibrate intrinsic and extrinsic camera parameters together?

Yes - this is the default, the same images will be used for both intrinsic and extrinsic calibration. See below for how to do separate intrinsic and extrinsic calibration.

### Can multical calibrate intrinsic parameters separately?

Yes - first calibrate separate intrinsic-only calibration (with images not needing to be corresponding), this will produce a calibration per camera in `intrinsic.json`.

`multical intrinsic --input_path intrinsic_images` 

Extrinsic-only calibration can then be performed using the known intrinsic parameters with `--calibration` to specify a prior intrinsic calibration to use, combined with `--fix_intrinsic` in order to avoid adjusting those parameters.

`multical calibrate --input_path extrinsic_images --calibration intrinsic_images/intrinsic.json --fix_intrinsic`


### How can I diagnose a bad calibration?

* Check that the boards are detected as expected (see above section on Calibration targets), in particular make sure the dimensions are correct for the pattern (e.g. 8x6 vs. 6x8)

* Ensure a correct camera model is being used. Note that there are currently no fisheye camera models - feel free to add one!
* Visualize the result and check that it matches expectations - look for patterns in the errors, is it just with particular frames or cameras? Check the initialization and verify if it is a problem with detecting the pattern or if the bundle adjustment step is causing the issue.

* Ensure input images are synchronized properly (images not captured at the same time will not calibrate well), or appropriate measures are taken to keep cameras still - for example a tripod.


### How can I evaluate the accuracy of a calibration?

* Reprojection error evaluates how well the models can fit the data. A low reprojection error is a good calibration if sufficient quantity and variation of input images are used. If the inputs are too few, the camera models may not be constrained enough to produce a good calibration.

* Compare a calibration against a different image set captured with the same cameras. Fix camera parameters with `--fix_intrinsic` and `--fix_camera_poses` and calibrate on the alternative image set:

`multical calibrate --input_path alternative_images --calibration calibration.json --fix_intrinsic --fix_camera_poses`

If the camera parameters and poses don't match the alternative image set well, the reprojection error will be high.

## Credits

Multical derives much inspiration from the [CALICO](https://github.com/amy-tabb/calico) application, which implements largely the algorithm presented in the paper "Calibration of Asynchronous Camera Networks: CALICO."

Tasnim Tabassum Nova, for her contribution to adding the non-overlapping camera case and making robustness improvements.

A number of abstractions and ideas found in [Anipose lib](https://github.com/lambdaloop/aniposelib) have been very useful and expanded upon. Small snippets of code have been used to initialise relative poses, which proved more robust in most cases than the least-squares method used in CALICO.

As with aniposelib, the scipy nonlinear optimizer [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) forms the basis for the bundle adjustment algorithm in this work. 

OpenCV provides many useful algorithms, which are used heavily here, for detecting calibration boards, initializing camera parameters, solving hand-eye calibration and modelling camera lens distortion.



## Authors

Oliver Batchelor
oliver.batchelor@canterbury.ac.nz

Tasnim Tabassum Nova
@TabassumNova
