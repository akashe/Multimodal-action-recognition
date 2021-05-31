import torch
import torch.nn as nn
import yaml
import argparse
import logging
from utils import get_Dataset_and_vocabs, collate_fn, train, evaluate, initialize_weights, count_parameters, epoch_time
from utils import add_to_writer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# runner loop
def runner(model, valid_iterator):

    model.eval()

    valid_loss, valid_stats = evaluate(model, valid_iterator)

    logger.info(f'\t Val. Loss: {valid_loss:.3f}')
    logger.info(f'\tValid f1: {valid_stats[0]:.3f}')
    logger.info(f'\tValid action accuracy: {valid_stats[1]:.3f}')
    logger.info(f'\t Valid object accuracy: {valid_stats[2]:.3f}')
    logger.info(f'\t Valid location accuracy: {valid_stats[3]:.3f}')


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="location to config yaml file")
    args = parser.parse_args()

    # Loading config:
    logger.info("Loading config")
    with open(args.config) as file:
        config = yaml.safe_load(file)
        logger.info(config)

    # Setting up Tensorboard
    writer = SummaryWriter('runs/v1')

    # Loading data:
    logger.info("Loading data and dataloaders")
    train_dataset, valid_dataset, vocab = get_Dataset_and_vocabs(config['data']['path'], \
                                                                 config['data']['train_file'], \
                                                                 config['data']['valid_file'], \
                                                                 config['data']['wavs_location'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn,
                                  num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2 * config['batch_size'], shuffle=True,
                                  collate_fn=collate_fn, num_workers=4)

    # checking device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'available device {device}')

    # Loading model
    model = nn.DataParallel(Model(audio_split_samples=config["audio_split_samples"], \
                                  hid_dim=config["hid_dim"], \
                                  audio_representation_layers=config["audio_representation_layers"], \
                                  n_heads=config["n_heads"], \
                                  pf_dim=config["pf_dim"], \
                                  dropout=config["dropout"], \
                                  max_length=config["positional_encoding_max_len"], \
                                  len_text_vocab=len(vocab['text_vocab']), \
                                  text_pad_index=vocab['text_vocab']['<pad>'], \
                                  text_representation_layers=config["text_representation_layers"], \
                                  cross_attention_layers=config["cross_attention_layers"], \
                                  output_dim_1=len(vocab['action_vocab']), \
                                  output_dim_2=len(vocab['object_vocab']), \
                                  output_dim_3=len(vocab['position_vocab']), \
                                  )).to(device)
    logger.info(f'Model loaded')

    # Initializing weights in model:
    model.apply(initialize_weights)

    # Number of model parameters
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'], betas=(config['optimizer']['beta1'],config['optimizer']['beta2']))

    runner(model, valid_dataloader)


if __name__ == "__main__":
    main()
