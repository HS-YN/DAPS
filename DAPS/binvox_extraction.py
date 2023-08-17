import os
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

import trimesh
# must be in appropriate location
from libmesh import check_mesh_contains

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="/home/directory/daps", type=str)
parser.add_argument("--voxelizer_path", default="/voxelizer/path/build/cuda_voxelizer", type=str)
args = parser.parse_args()

padding = 0.1
num_points = 100000
resolution_list = [16, 32]

data_path = args.data_path
subsample_path = os.path.join(data_path, 'scene_subsample_obj')

# cuda voxelizer (modify to your cuda_voxelizer path)
executor = args.voxelizer_path

for resolution in resolution_list:
    save_path = os.path.join(data_path, 'resolution_{:d}'.format(resolution))
    for scene in tqdm(os.listdir(subsample_path)):
        scene_path = os.path.join(subsample_path, scene)
        for obj in sorted(os.listdir(scene_path)):
            final_data_path = os.path.join(save_path, scene, str(obj)[:-4])
            os.makedirs(final_data_path)
            input_path = os.path.join(data_path, 'scene_subsample_obj/{:s}/{:s}'.format(scene, obj))
            output_path = os.path.join(data_path, 'scene_subsample_obj/{:s}/{:s}_{:d}.binvox'.format(scene, obj, resolution))
            move_path = os.path.join(final_data_path, 'model.binvox')
            command1 = '{:s} -f {:s} -s {:d} -o binvox'.format(executor, input_path, resolution)
            command2 = 'mv {:s} {:s}'.format(output_path, move_path)
            subprocess.run([command1], shell=True)
            subprocess.run([command2], shell=True)

            with open(move_path, 'rb') as f:
                voxel = trimesh.exchange.binvox.load_binvox(f, axis_order='xzy')
            mesh = voxel.marching_cubes
            # normalize mesh
            mesh.vertices += 0.5
            mesh.vertices /= resolution

            # sample pointcloud and normals from surface of the mesh
            pointcloud, idx = mesh.sample(num_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]

            point_iou = ((np.random.rand(num_points, 3).astype(np.float32)-0.5) * (1 + padding)) + 0.5
            occ = check_mesh_contains(mesh, point_iou)

            output_point_path = os.path.join(final_data_path, 'points.npz')
            np.savez(output_point_path, points=point_iou, occupancies=np.packbits(occ))
            output_pointcloud_path = os.path.join(final_data_path, 'pointcloud.npz')
            np.savez(output_pointcloud_path, pointcloud=pointcloud, normal=normals)
