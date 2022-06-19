import copy
import random
import numpy as np
import pyvista as pv
import open3d as o3d

skull_number = 3
model_number = 2

sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(skull_number)
targetPC_path = '../target_point_clouds/Optical/points1/sk{}/reg_pc{}.txt'.format(skull_number, model_number)


def square_distance(src, dst):
    return np.sum((src[:, None, :] - dst[None, :, :]) ** 2, axis=-1)


def calc_cd() -> float:

    sourcePC_mesh = pv.read(sourcePC_path)

    targetPC = np.loadtxt(targetPC_path)
    targetPC_points = pv.PolyData(targetPC)
    targetPC_mesh = targetPC_points.delaunay_2d()

    src_clean = np.asarray(sourcePC_mesh.points)
    ref_clean = np.asarray(targetPC_mesh.points)
    # ref_clean = copy.deepcopy(src_clean)

    while len(ref_clean) < 1600:
        ref_clean = np.concatenate([ref_clean, ref_clean])

    while len(ref_clean) > 1600:
        r_index = random.randint(0, len(ref_clean) - 1)
        ref_clean = np.delete(ref_clean, r_index, 0)

    # p_dists = square_distance(ref_clean, src_clean)
    #
    # src_matches = src_clean[np.array(
    #     [np.where(np.min(p_dists, axis=-1)[i] == p_dists[i]) for i in range(len(ref_clean))]
    # ).flatten()]

    while len(src_clean) < 1600:
        src_clean = np.concatenate([src_clean, src_clean])

    while len(src_clean) > 1600:
        r_index = random.randint(0, len(src_clean) - 1)
        src_clean = np.delete(src_clean, r_index, 0)

    dist_src = np.min(square_distance(src_clean, ref_clean), axis=-1)
    dist_ref = np.min(square_distance(ref_clean, src_clean), axis=-1)
    # chamfer_dist = np.mean(dist_src, axis=0) + np.mean(dist_ref, axis=0)

    chamfer_dist = np.mean(dist_ref)

    print(chamfer_dist)

    # p = pv.Plotter()
    #
    # p.add_points(src_matches, color='blue')
    # p.add_points(ref_clean, color='red')
    #
    # p.show()

    return float(chamfer_dist)

def main():

    calc_cd()

    # cds = np.zeros(10, dtype=np.float64)
    #
    # for i in range(10):
    #     cds[i] = calc_cd()
    #
    # print(np.mean(cds, dtype=np.float64))

if __name__ == "__main__":
    main()