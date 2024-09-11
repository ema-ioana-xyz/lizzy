# My Bachelor's Thesis Project
There are two parts to this project:

## 1. Evaluating three different _surface normal predictors_: [Three Filters To Normal](https://github.com/ruirangerfan/Three-Filters-to-Normal), [Surface Normal Uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty), and a module I made inspired by the depth-to-normal refinement in [GeoNet](https://github.com/xjqi/GeoNet)

### And here is my full paper for more details: [Lizzy.pdf](https://github.com/user-attachments/files/16969737/Lizzy.pdf)

~~The paper is mostly finished, save for the introduction; I will update it once the final version is done.~~
The final version of the paper was uploaded (2024-09-11)

The error metrics I used are the following:

$$ \text{Angle error} = \varphi = \arccos{\frac{y \cdot \hat{y}}{\|y\| \|\hat{y}\|} } $$

$$ \text{Mean error} =  \frac{1}{n}   \sum \varphi $$

$$ \text{Median error} =  \text{Median}(\varphi) $$

$$ \text{RMS error} = \left[ \frac{1}{n}   \sum \varphi^2   \right]^\frac{1}{2} $$

The test data (adapted from GeoNet's) is available on [HuggingFace](https://huggingface.co/datasets/ema-ioana-xyz/nyu-v2-geonet-test)

### _... And here are the results_
![image](https://github.com/user-attachments/assets/7a7986a3-caf7-4d2a-80cc-b9bd7c81977b)
![image](https://github.com/user-attachments/assets/67cc0126-7bfd-4275-83e4-fb44501718fe)


## 2. A demo web app that allows you to run predictions on image and video files while using these models
![flo-img-5](https://github.com/user-attachments/assets/098cedbb-3e27-4d2c-8a5d-11ae8ab5ca1e)
![flo-vid-7](https://github.com/user-attachments/assets/29eb19e6-5006-41b7-9917-f93cdfdcfc5e)
