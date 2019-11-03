'''
Function:
	train the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import argparse
import torch.nn as nn
from cfgs import cfg_train as cfg
from modules.utils.utils import *
from modules.models.dancenet import DanceNet


'''parse arguments for training'''
def parseArgs():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--videopath', dest='videopath', help='videopath for yielding training images', default='', type=str)
	args = parser.parse_args()
	return args


'''train'''
def train(cfg):
	# base prepare
	checkDir(cfg.BACKUP_DIR)
	logger_handle = Logger(cfg.LOGFILEPATH)
	args = parseArgs()
	if args.videopath:
		logger_handle.info('Start to generate images for training.')
		extractImagesFromVideo(args.videopath, logger_handle=logger_handle, savedir=cfg.ROOTDIR, target_imgsize=cfg.IMAGE_SIZE)
		logger_handle.info('Finish generating images for training!')
	use_cuda = torch.cuda.is_available()
	# prepare dataset
	dataset = ImageFolder(cfg.ROOTDIR, cfg.IMAGE_SIZE)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
	# prepare model
	model = DanceNet(image_size=cfg.IMAGE_SIZE)
	if use_cuda:
		model = model.cuda()
	model.initModules()
	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
	mse_loss_func = nn.MSELoss(size_average=False)
	# start to train
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	logger_handle.info('=' * 25 + 'TRAIN' + '=' * 25)
	logger_handle.info('DATASET is from %s, SIZE is %d...' % (cfg.ROOTDIR, len(dataset)))
	logger_handle.info('TRAIN EPOCHS: %d, BATCHSIZE: %d, BATCHS/EPOCH: %d...' % (cfg.MAX_EPOCHS, cfg.BATCH_SIZE, len(dataloader)))
	logger_handle.info('OPTIMIZER: Adam, LEARNINGRATE: %f...' % cfg.LEARNING_RATE)
	logger_handle.info('=' * 25 + 'TRAIN' + '=' * 25)
	for epoch in range(1, cfg.MAX_EPOCHS+1):
		logger_handle.info('Start epoch %d....' % epoch)
		for batch_idx, img in enumerate(dataloader):
			optimizer.zero_grad()
			img = img.type(FloatTensor)
			img_gen, mean, logvar = model(img)
			mse_loss = mse_loss_func(img_gen, img)
			kl_loss = torch.sum(mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
			loss = mse_loss + kl_loss
			loss.backward()
			optimizer.step()
			logger_handle.info('EPOCH: %d, BATCHS: %d/%d, LOSS: mse_loss %.3f, kl_loss %.3f, loss_total %.3f...' % (epoch, batch_idx, len(dataloader), mse_loss.item(), kl_loss.item(), loss.item()))
		if (epoch % cfg.SAVE_INTERVAL == 0) or (epoch == cfg.MAX_EPOCHS):
			savepath = os.path.join(cfg.BACKUP_DIR, 'epoch_%s.pth' % epoch)
			Logger.info('Saving checkpoints in %s...' % savepath)
			saveCheckpoints(model, savepath)


'''run'''
if __name__ == '__main__':
	train(cfg)