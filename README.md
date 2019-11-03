# DanceNet
```
generate dancing videos using vae.
```

# Introduction in Chinese
https://mp.weixin.qq.com/s/fFqztmu8hk5Jje9EUrP8DQ

# Environment
```
OS: Ubuntu16.04
Graphics card: Titan xp
cuda version: 10.0.130
cudnn version: 7.0.4
python version: 3.5+ with following dependencies:
	--pytorch==0.4.1
	--torchvision==0.2.2
	--numpy
	--argparse
	--opencv-python
```

# Usage
## Train
#### Step1
```
Modify the cfgs/cfg_train.py according to your needs.
```
#### Step2
```
Download the video for training following the introduction in the datasets dir.
```
#### Step3
```
Run train.py, command format is as follows:
python train.py --videopath datasets/videoname.mp4
```
## Demo
#### Step1
```
Modify the cfgs/cfg_demo.py according to your needs.
```
#### Step2
```
For generating dancing video randomly, run as follows:
python demo.py --mode random --checkpointspath epoch_50.pth --outputpath output.avi
For generating dancing video using trainset images, run as follows:
python demo.py --mode fromtrain --checkpointspath epoch_50.pth --outputpath output.avi
```

# Some results
#### fromtrain mode
![giphy](material/effects/demo_fromtrain.gif)
#### random mode
![giphy](material/effects/demo_random.gif)

# More
#### WeChat Official Accounts
*Charles_pikachu*  
![img](./material/pikachu.jpg)