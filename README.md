# HoloNav DL Registration

This Project Contains repositories used in evaluating HoloNav data on DL models Overlap PREDATOR by Huang et al., and RPMNet by Yew and Li.

The Project consists of 2 DL models modified to accept HoloNav input, as well as the HoloNav input data, located in the directory /surface_registration.

The surface registration directory in addition contains scripts to determine parameters for the DL models.

To evaluate PREDATOR, holonav.yaml can be used.

To evaluate RPMNet, please run
```
python eval.py --dataset_type holonav --dataset_path ../datasets/holonav --noise_type match --num_neighbors 20 --radius 12.0--rot_mag 45 -b 1 --resume ../logs/model-best/model-best.pth
```
