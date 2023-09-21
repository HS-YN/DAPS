# DAPS Benchmark

## Installation

### Download Matterport3d

1. Get Matterport3d dataset download code from the [link](https://niessner.github.io/Matterport/)
2. For those who use python 3, modify the download-mp.py
```python
# line 9
import urllib.request as urllib
# line 57
scan_id = scan_line.decode('utf8').strip('\n')
# line 73
print('\t' + url + ' > ' + out_file)
# line 132
key = input('')
# line 146
key = input('Press any key to continue on to main dataset download, or CTRL-C to exit.')
# line 170
key = input('')
```
3. Download Matterport3d habitat dataset
```python
mkdir daps  # in your dataset directory
python download-mp.py -o /home/directory/daps --task_data 'habitat'
cd /home/directory/daps/v1/tasks && unzip mp3d_habitat.zip
```

### Install Dependencies

1. Create Conda environment
- For Ubuntu <= 16.04, we recommend separating environments for image extraction(py3.9) and voxel extraction(py3.6).
```python
conda create -n daps python=3.9 cmake=3.14.0
conda activate daps
# for image extraction
conda install habitat-sim headless -c conda-forge -c aihabitat

# for voxel extraction
pip install open3d
pip install trimesh tqdm scikit-image
```
2. Download and install cuda voxelizer from the following [link](https://github.com/Forceflow/cuda_voxelizer)

### Create DAPS Dataset

For each file, configure the data and cuda voxelizer path
1. Image extraction
```python
python img_extraction.py --data_path=/home/directory/daps
```
2. Voxel extraction
```python
python mesh_extraction.py --data_path=/home/directory/daps
python binvox_extraction.py --data_path=/home/directory/daps --voxelizer_path=/voxelizer/path/bin/cuda_voxelizer
```

## Acknowledgement

The code for `libmesh` of voxel extraction is from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) (MIT License).
