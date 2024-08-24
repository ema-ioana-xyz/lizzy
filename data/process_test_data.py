from pathlib import Path
from h5py import File
from scipy.io import loadmat, savemat


OUTPUT_PATH = Path("/home/eit/mnt/ema-ioana-tit/geonet_nyu/GeoNet/test_data")
TEST_SPLIT = Path("/home/eit/mnt/ema-ioana-tit/geonet_nyu/GeoNet/gt/test_split.txt")
INPUT_ROOT = Path("/home/eit/mnt/ema-ioana-tit/geonet_nyu/GeoNet/gt/")


with open(TEST_SPLIT) as file:
    filenames = [name.rstrip() for name in file.readlines()]

images = loadmat(INPUT_ROOT / "images.mat")["images"]

# HDF5
depth = File(INPUT_ROOT / "depth.mat", "r")["depths"]
normals = File(INPUT_ROOT / "normals.mat", "r")["norm_gt_l"]
masks = File(INPUT_ROOT / "masks.mat", "r")["masks"]

for file in filenames:
    print(f"Processing {file}")
    index = int(file) - 1
    output_name = OUTPUT_PATH / f"{file}.mat"

    image = images[..., index]
    depth_val = depth[index]
    normals_val = normals[index]
    mask_val = masks[index]

    data = {
        "img": image,
        "depth": depth_val,
        "norm": normals_val,
        "mask": mask_val
    }
    savemat(output_name, data)
