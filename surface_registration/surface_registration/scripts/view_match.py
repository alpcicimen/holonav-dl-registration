import numpy as np
import pyvista as pv

RPMNet = False

def transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed

def inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    if g.shape[-2] == 4:
        inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform

def main():
    pred_transforms = np.load('../source_point_clouds_preop_models/pred_transforms.npy')
    gt_transforms = np.load('../source_point_clouds_preop_models/gt_transforms.npy', allow_pickle=True)

    if RPMNet:
        pred_transforms = pred_transforms[:, 4, ...]
        gt_transforms = gt_transforms[:, 0, ...]

    init_transform = inverse(gt_transforms)

    for i in range(0,len(pred_transforms)):

        if RPMNet:
            sel = 3 if i >= 8 else (2 if i >= 5 else 1)
        else:
            sel = 1

        sourcePC_path = '../source_point_clouds_preop_models/sk{}_face.ply'.format(sel)
        sourcePC_mesh = pv.read(sourcePC_path)
        # targetPC_path = '../target_point_clouds/Optical/points1/sk{}/reg_pc{}.txt'.format(1, i+1)
        #
        # targetPC = np.loadtxt(targetPC_path)
        # targetPC_points = pv.PolyData(targetPC)
        # targetPC_mesh = targetPC_points.delaunay_2d()

        targetPC_path = '../target_point_clouds/depthSensor/p1_example_cleaned.ply'
        targetPC_points = pv.read(targetPC_path)
        targetPC_mesh = targetPC_points.delaunay_2d()

        src_transformed = np.asarray(sourcePC_mesh.points)
        src_transformed = np.array(transform(init_transform[i], src_transformed))

        p = pv.Plotter()

        p.add_points(targetPC_mesh, smooth_shading=True, color='green')

        # p.add_mesh(sourcePC_mesh, smooth_shading=True, opacity=0.3)
        # p.add_points(src_transformed, smooth_shading=True, color='red')

        src_transformed = np.array(transform(pred_transforms[i], src_transformed))

        # p.add_points(src_transformed, smooth_shading=True, color='blue')

        sourcePC_mesh.points = src_transformed

        p.add_mesh(sourcePC_mesh, smooth_shading=True, opacity=0.3)

        p.show()


if __name__ == '__main__':
    main()