'''
Function:
	define some utils.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cv2
import torch
import logging
import numpy as np
from torch.utils.data import Dataset


'''checkdir'''
def checkDir(dirpath):
	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
		return False
	return True


'''log function.'''
class Logger():
	def __init__(self, logfilepath, **kwargs):
		logging.basicConfig(level=logging.INFO,
							format='%(asctime)s %(levelname)-8s %(message)s',
							datefmt='%Y-%m-%d %H:%M:%S',
							handlers=[logging.FileHandler(logfilepath),
									  logging.StreamHandler()])
	@staticmethod
	def log(level, message):
		logging.log(level, message)
	@staticmethod
	def debug(message):
		Logger.log(logging.DEBUG, message)
	@staticmethod
	def info(message):
		Logger.log(logging.INFO, message)
	@staticmethod
	def warning(message):
		Logger.log(logging.WARNING, message)
	@staticmethod
	def error(message):
		Logger.log(logging.ERROR, message)


'''extract images from a video'''
def extractImagesFromVideo(videopath, logger_handle, savedir='datasets/images', frame_interval=3, target_imgsize=(208, 120), bg_thresh=20):
	checkDir(savedir)
	capture = cv2.VideoCapture(videopath)
	count = 0
	while capture.isOpened():
		ret, img = capture.read()
		if ret:
			count += 1
			if count % frame_interval == 0:
				img = cv2.resize(img, target_imgsize)
				img[np.where(np.greater(img, bg_thresh))] = 255
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				savepath = os.path.join(savedir, str(len(os.listdir(savedir))+1)+'.jpg')
				logger_handle.info('save one image in %s...' % savepath)
				cv2.imwrite(savepath, img)
		else:
			break


'''save checkpoints'''
def saveCheckpoints(model, savepath):
	torch.save(model.state_dict(), savepath)
	return True


'''load checkpoints'''
def loadCheckpoints(model, checkpointspath):
	model.load_state_dict(torch.load(checkpointspath))
	return model


'''load data'''
class ImageFolder(Dataset):
	def __init__(self, rootdir, image_size, **kwargs):
		self.image_size = image_size
		self.imagepaths = [os.path.join(rootdir, str(i)+'.jpg') for i in range(1, len(os.listdir(rootdir))+1)]
	def __getitem__(self, index):
		img = cv2.imread(self.imagepaths[index], cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, self.image_size)
		img = img.astype(np.float32) / 255
		img = torch.from_numpy(img).unsqueeze(-1).permute(2, 0, 1)
		return img
	def __len__(self):
		return len(self.imagepaths)