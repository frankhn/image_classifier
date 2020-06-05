import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', action='store', default='3')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    parser.add_argument('--gpu', action="store", default="gpu")
    
    return parser.parse_args()