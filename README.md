# Simple RGB-D SLAM

Simple RGB-D SLAM Implementation for Research

![ezgif com-gif-maker](https://user-images.githubusercontent.com/38591115/158515838-904e531b-7d5b-45fc-9b31-103a468827a1.gif)

## Dependencies
Dependencies are listed in the table below along with the version used during development and testing.
| Dependency    | Tested Version |
| :---:         | :---:  |
| OpenCV        | 4.5.5  |
| Ceres         | 2.0.0  |
| Sophus        | 1.0.0  |
| Eigen         |3.3.4|
| CSparse       |5.1.2|
| Pangolin      |0.5|
| Glog          |0.3.5|
| DBoW3          |1.0|

## Build
```Bash
git clone https://github.com/93won/RGBD_SLAM && cd RGBD_SLAM
mkdir build && cd build
make
```

## Example
### TUM Dataset
[1] Download a 'freiburg2_desk' sequence from https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz and uncompress it. <\br>
[2] Execute associate.py in the script folder as follows to create the associations.txt file.
```Bash
python associate.py DATA_FOLDER/rgb.txt DATA_FOLTER/depth.txt
```
