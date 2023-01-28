POP_PATH = 'data/POP909/'
CLASSIC_PATH = 'data/maestro-v3.0.0/'
SCORES_PATH = 'data/mydata/scores.cmp'
SPLIT_DATA_PATH = 'data/mydata/split_data.cmp'
INPUT_NOTE_NUM = 256

# TRAIN_PARAMETER
DROP_RATE=0.3
OPTIMIZER='AdamW'
CRITERION='BCE'
LEARNING_RATE=3e-5
BATCH_SIZE=8
DEVICE='cpu'
PATIENCE=7
MODEL_SAVE_PATH='model/model.cpt'
LOG_FILE='model/log'