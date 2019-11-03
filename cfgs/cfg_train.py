'''config file for train'''


IMAGE_SIZE = (208, 120) # W x H
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 4
ROOTDIR = 'datasets/images'
BACKUP_DIR = 'backup'
LOGFILEPATH = 'backup/train.log'
SAVE_INTERVAL = 5