# DanceNet
```
generate dancing videos using vae.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```

# Introduction
#### in Chinese
https://mp.weixin.qq.com/s/fFqztmu8hk5Jje9EUrP8DQ

# Environment
```
OS: Ubuntu16.04
Graphics card: Titan xp
Python: Python3.5+(have installed the neccessary dependencies)
```

# Usage
#### Train
```
usage: train.py [-h] [--videopath VIDEOPATH]

optional arguments:
  -h, --help            show this help message and exit
  --videopath VIDEOPATH
                        videopath for yielding training images

cmd example:
python train.py --videopath datasets/videoname.mp4
```
#### Test
```
usage: demo.py [-h] --mode MODE --checkpointspath CHECKPOINTSPATH
               [--outputpath OUTPUTPATH]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           mode for yielding dancing video, support <random> or
                        <fromtrain>...
  --checkpointspath CHECKPOINTSPATH
                        checkpoints path
  --outputpath OUTPUTPATH
                        output path

cmd example:
python demo.py --mode random --checkpointspath epoch_50.pth --outputpath output.avi
python demo.py --mode fromtrain --checkpointspath epoch_50.pth --outputpath output.avi
```

# Results
#### fromtrain mode
![giphy](docs/effects/demo_fromtrain.gif)
#### random mode
![giphy](docs/effects/demo_random.gif)

# More
#### WeChat Official Accounts
*Charles_pikachu*  
![img](./docs/pikachu.jpg)