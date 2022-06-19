import numpy as np
import os
import pyvista as pv
import open3d as o3d


if __name__ == '__main__':

    source_clouds = []

    sk_path = '../source_point_clouds_preop_models'
    reg_path = '../target_point_clouds/Optical/points1/sk1'
    sk1_regs = [np.loadtxt(os.path.join(reg_path, fname)) for fname in os.listdir(reg_path) if 'reg_' in fname]


    reg_path = '../target_point_clouds/Optical/points1/sk2'
    sk2_regs = [np.loadtxt(os.path.join(reg_path, fname)) for fname in os.listdir(reg_path) if 'reg_' in fname]

    reg_path = '../target_point_clouds/Optical/points1/sk3'
    sk3_regs = [np.loadtxt(os.path.join(reg_path, fname)) for fname in os.listdir(reg_path) if 'reg_' in fname]

    all_regs = sk1_regs + sk2_regs + sk3_regs

    print('great success')
