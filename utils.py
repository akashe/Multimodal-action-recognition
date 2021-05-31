import os.path
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import spacy
import logging
from collections import Counter
import csv
import pickle
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

logger.info("----Loading Spacy----")
spacy_en = spacy.load('en')


# Calculate F1: use scikit learn and use weighted and use
# data like this https://stackoverflow.com/questions/46732881/how-to-calculate-f1-score-for-multilabel-classification
# get statistics


def get_f1(y_pred, y_label):
    # a_pred = a_pred.detach().cpu().tolist() # action predicted
    # o_pred = o_pred.detach().cpu().tolist() # objects predicted
    # p_pred = p_pred.detach().cpu().tolist() # positions predicted
    #
    # a_label = a_label.detach().cpu().tolist() # true actions
    # o_label = o_label.detach().cpu().tolist() # true objects
    # p_label = p_label.detach().cpu().tolist() # true positions

    y_pred = np.array(y_pred)
    y_true = np.array(y_label)

    return f1_score(y_true=y_true, y_pred=y_pred, average="weighted")


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def numericalize(inputs, vocab=None, tokenize=False):
    # Create vocabs for train file
    if vocab == None:
        # check unique tokens
        counter = Counter()
        for i in inputs:
            if tokenize:
                counter.update(tokenizer(i))
            else:
                counter.update([i])

        # Create Vocab
        if tokenize:  # That is we are dealing with sentences.
            vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        else:
            vocab = {}

        vocab.update({j: i + len(vocab) for i, j in enumerate(counter)})

    # Convert tokens to numbers:
    numericalized_inputs = []
    for i in inputs:
        if tokenize:
            numericalized_inputs.append([vocab[j] if i in vocab else vocab["<unk>"] for j in
                                         tokenizer(i)])  # TODO: doing tokenization twice here
        else:
            numericalized_inputs.append(vocab[i])

    return numericalized_inputs, vocab


def collate_fn(batch, device, text_pad_value, audio_pad_value):
    """
    We use this function to pad the inputs so that they are of uniform length
    and convert them to tensor

    Note: This function would fail while using Iterable type dataset because
    while calculating max lengths the items in iterator will vanish.
    """
    max_audio_len = 0
    max_text_len = 0

    batch_size = len(batch)

    for audio_clip, transcript, _, _, _ in batch:
        if len(audio_clip) > max_audio_len:
            max_audio_len = len(audio_clip)
        if len(transcript) > max_text_len:
            max_text_len = len(transcript)

    audio = torch.FloatTensor(batch_size, max_audio_len).fill_(audio_pad_value).to(device)
    text = torch.IntTensor(batch_size, max_text_len).fill_(text_pad_value).to(device)
    action = torch.IntTensor(batch_size).fill_(0).to(device)
    object_ = torch.IntTensor(batch_size).fill_(0).to(device)
    position = torch.IntTensor(batch_size).fill_(0).to(device)

    for i, (audio_clip, transcript, action_taken, object_chosen, position_chosen) in enumerate(batch):
        audio[i][:len(audio_clip)] = audio_clip
        text[i][:len(transcript)] = transcript
        action[i] = action_taken
        object_[i] = object_chosen
        position[i] = position_chosen

    return audio, text, action, object_, position


class Dataset:
    def __init__(self, audio, text, action, object_, position, wavs_location):
        self.audio = audio
        self.text = text
        self.action = action
        self.object = object_
        self.position = position
        self.wavs_location = wavs_location

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        _, wav = wavfile.read(self.wavs_location + self.audio[item])
        return wav, self.text[item], self.action[item], self.object[item], self.position[item]


def load_csv(path, file_name):
    # Loads a csv and returns columns:
    with open(os.path.join(path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        au, t, ac, o, p = [], [], [], [], []
        for row in csv_reader:
            au.append(row[0])
            t.append(row[1])
            ac.append(row[2])
            o.append(row[3])
            p.append(row[4])
    return au, t, ac, o, p


def get_Dataset_and_vocabs(path, train_file_name, valid_file_name, wavs_location):
    train_data = load_csv(path, train_file_name)
    test_data = load_csv(path, valid_file_name)

    vocabs = []  # to store all the vocabs
    # audio location need not to be numercalized
    # audio files will be loaded in Dataset.__getitem__()
    numericalized_train_data = [train_data[0]]  # to store train data after converting string to ints
    numericalized_test_data = [test_data[0]]  # to store test data after converting strings to ints

    for i, (j, k) in enumerate(zip(train_data[1:], test_data[1:])):
        if i == 0:  # We have to only tokenize transcripts which come at 0th position
            a, vocab = numericalize(j, tokenize=True)
            b, _ = numericalize(k, vocab=vocab, tokenize=True)
        else:
            a, vocab = numericalize(j)
            b, _ = numericalize(k, vocab=vocab)
        numericalized_train_data.append(a)
        numericalized_test_data.append(a)
        vocabs.append(vocab)

    train_dataset = Dataset(*numericalized_train_data, wavs_location)
    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    Vocab = {'text_vocab': vocabs[0], 'action_vocab': vocabs[1], 'object_vocab': vocabs[2], 'position_vocab': vocabs[3]}

    # dumping vocab
    with open(os.path.join(path, "vocab"), "wb") as f:
        pickle.dump(Vocab, f)

    return train_dataset, valid_dataset, Vocab


def get_Dataset_and_vocabs_for_eval(path, valid_file_name, wavs_location):
    test_data = load_csv(path, valid_file_name)

    with open(os.path.join(path, "vocab"), "wb") as f:
        Vocab = pickle.load(f)

    numericalized_test_data = [test_data[0], numericalize(test_data[1], vocab=Vocab['text_vocab'], tokenize=True)[0],
                               numericalize(test_data[2], vocab=Vocab['action_vocab'])[0],
                               numericalize(test_data[3], vocab=Vocab['object_vocab'])[0],
                               numericalize(test_data[4], vocab=Vocab['position_vocab'])[0]]

    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    return valid_dataset, Vocab


def initialize_weights(m):
    # if hasattr(m, 'weight') and m.weight.dim() > 1:
    #     nn.init.xavier_uniform_(m.weight.data)
    for name, param in m.named_parameters():
        if not isinstance(m, nn.Embedding):
            nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, optim, clip):
    model.train()

    epoch_loss = 0

    # Tracking accuracies
    action_accuracy = []
    object_accuracy = []
    position_accuracy = []

    # for f1
    y_pred = []
    y_true = []

    for i, batch in enumerate(train_iterator):
        # running batch
        train_result = model(batch)

        optim.zero_grad()

        loss = train_result["loss"]
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optim.step()

        # Statistics
        epoch_loss += loss.item()
        y_pred.append([train_result["predicted_action"].tolist(), train_result["predicted_action"].tolist(),
                       train_result["predicted_location"].tolist()])
        y_true.append([batch[2].tolist(), batch[3].tolist(), batch[4].tolist()])

        action_accuracy.append(sum(train_result["predicted_action"] == batch[2]) / len(batch[2]) * 100)
        object_accuracy.append(sum(train_result["predicted_action"] == batch[3]) / len(batch[2]) * 100)
        position_accuracy.append(sum(train_result["predicted_location"] == batch[4]) / len(batch[2]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_action_accuracy = sum(action_accuracy) / len(action_accuracy)
    epoch_object_accuracy = sum(object_accuracy) / len(object_accuracy)
    epoch_position_accuracy = sum(position_accuracy) / len(position_accuracy)

    return epoch_loss / len(
        train_iterator), (epoch_f1, epoch_action_accuracy, epoch_object_accuracy, epoch_position_accuracy)


def evaluate(model, valid_iterator):
    model.eval()

    epoch_loss = 0

    # Tracking accuracies
    action_accuracy = []
    object_accuracy = []
    position_accuracy = []

    # for f1
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            # running batch
            valid_result = model(batch)

            loss = valid_result["loss"]

            # Statistics
            epoch_loss += loss.item()

            y_pred.append([valid_result["predicted_action"].tolist(), valid_result["predicted_action"].tolist(),
                           valid_result["predicted_location"].tolist()])
            y_true.append([batch[2].tolist(), batch[3].tolist(), batch[4].tolist()])

            action_accuracy.append(sum(valid_result["predicted_action"] == batch[2]) / len(batch[2]) * 100)
            object_accuracy.append(sum(valid_result["predicted_action"] == batch[3]) / len(batch[2]) * 100)
            position_accuracy.append(sum(valid_result["predicted_location"] == batch[4]) / len(batch[2]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_action_accuracy = sum(action_accuracy) / len(action_accuracy)
    epoch_object_accuracy = sum(object_accuracy) / len(object_accuracy)
    epoch_position_accuracy = sum(position_accuracy) / len(position_accuracy)

    return epoch_loss / len(
        valid_iterator), (epoch_f1, epoch_action_accuracy, epoch_object_accuracy, epoch_position_accuracy)


def add_to_writer(writer,epoch,train_loss,valid_loss,train_stats,valid_stats):
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Validation loss", valid_loss, epoch)
    writer.add_scalar("Train f1", train_stats[0], epoch)
    writer.add_scalar("Train action accuracy", train_stats[1], epoch)
    writer.add_scalar("Train object accuracy", train_stats[2], epoch)
    writer.add_scalar("Train location accuracy", train_stats[3], epoch)
    writer.add_scalar("Valid f1", valid_stats[0], epoch)
    writer.add_scalar("Valid action accuracy", valid_stats[1], epoch)
    writer.add_scalar("Valid object accuracy", valid_stats[2], epoch)
    writer.add_scalar("Valid location accuracy", valid_stats[3], epoch)
