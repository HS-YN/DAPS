import open3d as o3d

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="/home/directory/daps", type=str)
args = parser.parse_args()

with open('scene_pos_coord_3d.pkl', 'rb') as g:
    scene_position_coord = pickle.load(g)
scene_position_coord = scene_position_coord[0]
scenes = list(scene_position_coord.keys())


save_path = args.data_path
for scene in tqdm(scenes):
    scene_path = os.path.join(save_path, 'v1/tasks/mp3d/{:s}/{:s}_semantic.ply'.format(scene, scene))
    mesh = o3d.io.read_triangle_mesh(scene_path)
    save_pc_path = os.path.join(save_path, 'scene_subsample_obj', scene)
    os.makedirs(save_pc_path)

    for node in scene_position_coord[scene].keys():
        save_pc_path_ply = os.path.join(save_pc_path, '{:d}.obj'.format(node))
        node_coord = scene_position_coord[scene][node]
        curr_coord = [node_coord[0], node_coord[1], node_coord[2]]

        # set cutting boundary for the scene
        all_x = [2.5*e for e in [-1, 1]]
        all_y = [2.5*e for e in [-1, 1]]
        all_z = [2*e for e in [-1, 1]]
        bbox_coord = np.repeat(np.expand_dims(np.array(curr_coord), axis=0), 8, axis=0)
        bbox_offset = np.array(np.meshgrid(all_x, all_y, all_z)).T.reshape(-1,3)
        bbox_coord += bbox_offset

        bbox_points = o3d.geometry.PointCloud()
        bbox_points.points = o3d.utility.Vector3dVector(bbox_coord)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points.points)
        cropped_mesh = mesh.crop(bbox)

        # remove small clusters
        tmp_mesh = cropped_mesh.cluster_connected_triangles()
        if np.argmax(tmp_mesh[1]) != np.argmax(tmp_mesh[2]):
            diff = abs(tmp_mesh[2][np.argmax(tmp_mesh[1])] - tmp_mesh[2][np.argmax(tmp_mesh[2])])
            if diff >= 35:
                cluster_idx = np.where(tmp_mesh[0] == np.argmax(tmp_mesh[2]))[0]
            else:
                cluster_idx1 = np.where(tmp_mesh[0] == np.argmax(tmp_mesh[1]))[0]
                cluster_idx2 = np.where(tmp_mesh[0] == np.argmax(tmp_mesh[2]))[0]
                cluster_idx = np.concatenate((cluster_idx1, cluster_idx2))
        else:
            cluster_idx = np.where(tmp_mesh[0] == np.argmax(tmp_mesh[2]))[0]
        triangle_idx = np.asarray(cropped_mesh.triangles)
        mesh_idx = np.unique(triangle_idx[cluster_idx])
        clustered_mesh = cropped_mesh.select_down_sample(mesh_idx)

        # open3d<=0.9.0 must manually remove mtl files (rm -rf */*/*.mtl)
        out = o3d.io.write_triangle_mesh(save_pc_path_ply, clustered_mesh, write_triangle_uvs=False)
