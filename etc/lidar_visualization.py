import numpy as np
import os
import open3d as o3d

file_dir = os.path.dirname(__file__)

def get_ply_from_npy(file_dir):
    gt_npy = np.load(os.path.join(file_dir, "polarimetric_imaging_dataset", "20220621_142942", "XYZ", "0001365.npz"))

    # print(gt_npy)
    xyz = gt_npy["xyz"]
    intensity = gt_npy["intensity"]
    print(np.shape(xyz))
    print(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("./data.ply", pcd) ## for saving result
    o3d.visualization.draw_geometries([pcd])

def get_ply_from_bin(file_dir):
    gt_bin = np.fromfile(os.path.join(file_dir, "sample_velo.bin"), dtype=np.float32).reshape(-1, 4)
    xyz = gt_bin[:, :3]

    print(np.shape(xyz))
    print(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("./data.ply", pcd) ## for saving result
    o3d.visualization.draw_geometries([pcd])


get_ply_from_npy(file_dir)
# get_ply_from_bin(file_dir)


