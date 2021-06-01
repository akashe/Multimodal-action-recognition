import torch
import torch.nn as nn
import yaml
import argparse
import logging
from functools import partial
from utils import get_Dataset_and_vocabs, collate_fn, train, evaluate, initialize_weights, count_parameters, epoch_time
from utils import add_to_writer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
import time, datetime, os,sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                              '%m-%d-%Y %H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)


file_handler = logging.FileHandler('logs/train_logs_'+str(datetime.datetime.now())[:-7]+'.log')
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


# runner loop
def runner(epochs, model, train_iterator, valid_iterator, optim, writer, config):
    clip, save_path, model_name = config["clip"], config['data']['path'], config['model_name']

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_stats = train(model, train_iterator, optim, clip)
        valid_loss, valid_stats = evaluate(model, valid_iterator)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_path , model_name))

        logger.info("-------------------------")
        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f}')
        logger.info(f'\t Val. Loss: {valid_loss:.3f}')
        logger.info(f'\t Train f1: {train_stats[0]} \n Valid f1: {valid_stats[0]}')
        logger.info(f'\t Train action accuracy: {train_stats[1]:.3f} \t Valid action accuracy: {valid_stats[1]:.3f}')
        logger.info(f'\t Train object accuracy: {train_stats[2]:.3f} \t Valid object accuracy: {valid_stats[2]:.3f}')
        logger.info(
            f'\t Train location accuracy: {train_stats[3]:.3f} \t Valid location accuracy: {valid_stats[3]:.3f}')

        add_to_writer(writer, epoch, train_loss, valid_loss, train_stats, valid_stats, config)

    # dumping config file
    with open(config['log_path'] + "/config.yaml", "w") as file:
        _ = yaml.dump(config, file)

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

    # Setting up Tensorboard
    config['log_path'] = os.path.join(config['log_path'], str(datetime.datetime.now())[:-7])
    writer = SummaryWriter(config['log_path'])

    # Loading data:
    logger.info("Loading data and dataloaders")
    train_dataset, valid_dataset, vocab = get_Dataset_and_vocabs(config['data']['path'], \
                                                                 config['data']['train_file'], \
                                                                 config['data']['valid_file'], \
                                                                 config['data']['wavs_location'])

    collate_fn_ = partial(collate_fn, device=device, text_pad_value=vocab['text_vocab']["<pad>"] \
                          , audio_pad_value=0, audio_split_samples=config["audio_split_samples"])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_,
                                  )
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
                  config=config \
                  ).to(device)
    logger.info(f'Model loaded')

    # Initializing weights in model:
    model.apply(initialize_weights)

    # Number of model parameters
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'],
                             betas=(config['optimizer']['beta1'], config['optimizer']['beta2']))

    runner(config["epochs"], model, train_dataloader, valid_dataloader, optim, writer, config)

    writer.close()


if __name__ == "__main__":
    main()
