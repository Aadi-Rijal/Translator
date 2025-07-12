import warnings
from config import get_config
from train import train_model

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)