import argparse
import datetime
import logging
import os
import sys
from functools import partial

import torch
import yaml
from torch.utils.data import DataLoader

from model import Model
from utils import get_Dataset_and_vocabs_for_eval, collate_fn, evaluate, count_parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)


file_handler = logging.FileHandler('logs/eval_logs_'+str(datetime.datetime.now())[:-7]+'.log')
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def runner(model, valid_iterator):
    model.eval()

    valid_loss, valid_stats = evaluate(model, valid_iterator)

    logger.info(f'\t Val. Loss: {valid_loss:.3f}')
    logger.info(f'\tValid f1: {valid_stats[0]}')
    logger.info(f'\tValid action accuracy: {valid_stats[1]:.3f}')
    logger.info(f'\t Valid object accuracy: {valid_stats[2]:.3f}')
    logger.info(f'\t Valid location accuracy: {valid_stats[3]:.3f}')


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="location to config yaml file")
    args = parser.parse_args()

    # checking device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'available device {device}')

    # Loading config:
    logger.info("Loading config")
    with open(args.config) as file:
        config = yaml.safe_load(file)
        for i in config:
            logger.info(f'{i} : {config[i]}')

    # Loading data:
    logger.info("Loading data and dataloaders")
    valid_dataset, vocab = get_Dataset_and_vocabs_for_eval(config['data']['path'], \
                                                           config['data']['valid_file'], \
                                                           config['data']['wavs_location'])

    collate_fn_ = partial(collate_fn, device=device, text_pad_value=vocab['text_vocab']["<pad>"] \
                          , audio_pad_value=0, audio_split_samples=config["audio_split_samples"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=2 * config['batch_size'], shuffle=True,
                                  collate_fn=collate_fn_)

    # Loading model
    model = Model(audio_split_samples=config["audio_split_samples"], \
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
                                config=config
                                  ).to(device)

    # Loading model weights in model:
    if device.type == 'cpu':
        model.load_state_dict(torch.load(os.path.join(config['data']['path'],config["model_name"]),map_location=torch.device('cpu')))
    else:
        model.load_state_dict(
            torch.load(os.path.join(config['data']['path'], config["model_name"])))
    logger.info(f'Model loaded')

    # Number of model parameters
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    runner(model, valid_dataloader)


if __name__ == "__main__":
    main()
