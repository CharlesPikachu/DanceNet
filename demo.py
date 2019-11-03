'''
Function:
	demo for test our model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cv2
import numpy as np
from cfgs import cfg_demo as cfg
from modules.utils.utils import *
from modules.models.dancenet import DanceNet


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--mode', dest='mode', help='mode for yielding dancing video, support <random> or <fromtrain>...', default='fromtrain', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints path', default='', type=str, required=True)
	parser.add_argument('--outputpath', dest='outputpath', help='output path', default='output.avi', type=str)
	args = parser.parse_args()
	return args


'''demo'''
def demo(cfg):
	args = parseArgs()
	# prepare
	logger_handle = Logger(cfg.LOGFILEPATH)
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	model = DanceNet(image_size=cfg.IMAGE_SIZE)
	if use_cuda:
		model = model.cuda()
	model = loadCheckpoints(model, args.checkpointspath)
	# generate video according to mode
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter(args.outputpath, fourcc, 30.0, cfg.IMAGE_SIZE)
	if args.mode == 'random':
		for i in range(1, cfg.NUM_SAMPLES+1):
			logger_handle.info('[RANDOM]: Yield %d image...' % i)
			z = np.random.normal(0, 1, 128).astype(np.float32)
			z = torch.from_numpy(z).view(1, -1).type(FloatTensor)
			img_gen = model.decoder(z)[0].cpu().data.permute(1, 2, 0).numpy() * 255
			img_gen = img_gen.astype('uint8')
			img_gen = cv2.cvtColor(img_gen, cv2.COLOR_GRAY2RGB)
			video.write(img_gen)
	elif args.mode == 'fromtrain':
		for i in range(1, cfg.NUM_SAMPLES+1):
			logger_handle.info('[FROMTRAIN]: Yield %d image...' % i)
			img = cv2.imread(os.path.join(cfg.ROOTDIR, '%d.jpg' % i), cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, cfg.IMAGE_SIZE)
			img = img.astype(np.float32) / 255
			img = torch.from_numpy(img).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
			img_gen = model(img)[0].cpu().data.permute(1, 2, 0).numpy() * 255
			img_gen = img_gen.astype('uint8')
			img_gen = cv2.cvtColor(img_gen, cv2.COLOR_GRAY2RGB)
			video.write(img_gen)
	else:
		raise ValueError('Unsupport args.mode <%s>...' % args.mode)
	video.release()


'''run'''
if __name__ == '__main__':
	demo(cfg)