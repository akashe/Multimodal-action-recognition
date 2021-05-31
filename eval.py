import torch
import torch.nn as nn
import yaml
import argparse
import logging
from utils import get_Dataset_and_vocabs_for_eval, collate_fn, train, evaluate, initialize_weights, count_parameters, epoch_time
from utils import add_to_writer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
import time

logger = logging.getLogger(__name__)

# runner loop
def runner(epochs, model, valid_iterator):

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_stats = train(model, train_iterator, optim, clip)
        valid_loss, valid_stats = evaluate(model, valid_iterator)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'data/multimodal-multilabel-model.pt')

        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f}')
        logger.info(f'\t Val. Loss: {valid_loss:.3f}')
        logger.info(f'\t Train f1: {train_stats[0]:.3f} \t Valid f1: {valid_stats[0]:.3f}')
        logger.info(f'\t Train action accuracy: {train_stats[1]:.3f} \t Valid action accuracy: {valid_stats[1]:.3f}')
        logger.info(f'\t Train object accuracy: {train_stats[2]:.3f} \t Valid object accuracy: {valid_stats[2]:.3f}')
        logger.info(
            f'\t Train location accuracy: {train_stats[3]:.3f} \t Valid location accuracy: {valid_stats[3]:.3f}')

        add_to_writer(writer, epoch, train_loss, valid_loss, train_stats, valid_stats)




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

    # Loading data:
    logger.info("Loading data and dataloaders")
    train_dataset, valid_dataset, vocab = get_Dataset_and_vocabs_for_eval(config['data']['path'], \
                                                                 config['data']['valid_file'], \
                                                                 config['data']['wavs_location'])

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

    # Loading model weights in model:
    model.load_state_dict(torch.load(config["saved_model"]))
    logger.info(f'Model loaded')

    # Number of model parameters
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    runner(config["epochs"], model, train_dataloader, valid_dataloader, optim, writer, config["clip"])




if __name__ == "__main__":
    main()