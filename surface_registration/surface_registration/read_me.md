# Surface registration - getting started

## Python libraries to install
- numpy
- pyvista
- scipy
- pandas
- matplotlib

## Getting started:
- you can start by unzipping the `modelnet40_ply_hdf5_2048.zip` and running `laod_plot_modelNet40.py` to visualize the data and get yourself familiar with it. ModelNet40 can be used for training the DL approach and to evaluate both the DL and ICP based approaches.
- run `laod_plot_modelNet40.py` to load and visualize our point cloud data. We have 3 skull models but for now data were acquired only for skull 1. you can change the `target_pc` parameters with the different options:
	- `optical_raw`: not registered target point clouds acquired from the optical tracking data
	- `optical_reg`: registered target point clouds (using landmarks) acquired from the optical tracking data
	- `pre-op`: sampled target point clouds from the preoperative model (source point cloud)

Change the number of the respective target point cloud `pc_number` to visualize the different samples of point clouds.


## Structure of folders:
- `source_point_cloud_preop_models`: contains the pre-operative models (source point clouds) we are interested to trasform.
- `target_point_cloud`: contains the target point clouds we want to transform towards. It has to subfolders `Optical` which contains point clouds acquired using the optical pointer, and `PreOp` contains point clouds sampled from the preoperative models.
- `modelnet40_ply_hdf5_2048.zip`: compressed file contains the ModelNet40 dataset, which can be used for training and first stage evaluation.
- `scripts`: contains simple python scripts to load and visualized point clouds, mainly `load_plot_point_clouds.py` to load source and target point clouds from our evalauation point clouds, and `laod_plot_modelNet40.py` to load and visualize ModelNet40 data. There is also another script `processAcquiredOpticalData.py` which was used to prepare and process the data acquired with the optical tracker.

