# Multimodal action recognition:

#### Task Definition
Given an audio snippet and a transcript, classify the referred 'action', 'object' and 'position/location' information present in them.

#### Set up virtual environment
To set up a python virtual environment with the required dependencies:
```
python3 -m venv multimodal_classification
source multimodal_classification/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --use-feature=2020-resolver
python -m spacy download en
```

#### Train & eval

Set the location of downloaded wav files in 'wavs_location' in config.yaml. You can also set location of custom train and eval files in config.yaml.

To train the model run
```
python train.py --config config.yaml
```
where, config.yaml consists the location of train. valid scripts, saved_model_location and other hyper-parameters.

To eval
```
python eval.py --config config.yaml
```
While evaluating, the model can accept csv files with their location mentioned in config file. It won't support single sentence inference because it would need corresponding audio sample also.

To eval with your own csv file. Copy the file in data folder and update the 'valid_file' name parameter in the config. 

#### Tensorboard & logs
The logs are present in 'logs/'. 

To visualize using tensorboard use event files in 'runs/'. The sub-folders in the 'runs/' folder are the experiment name which you set in config as 'log_path'

```
tensorboard --logdir path_to_tensorboard_logs
```
Tensorboard logs are present in config.log_path and the specific mode you are running training the model in

#### Model
We use a multimodal model here. The model consists of 3 main components:
1. **Audio self-attention**: These layers calculate self attention among the audio signals. We take the original audio len and split in equal parts controlled by the parameter audio_split_samples. So, if the original audio len was 60000 and audio_split_samples = 1000 then we divide the audio into 60 tokens.
2. **Text self-attention**: These layers find self attention in the text representations.
3. **Cross- attention**: After getting text and audio representation we find cross attention between them and use the results for prediction.

Each layer has the following sequence of operations:
1. Calculate attention. (Note: In case of cross-attention, we use audio representations as key and value values and use them to find attention over text representations which we set as query)
2. LayerNorm + residual connection
3. Pointwise Feedforward.
4. LayerNorm + residual connection.

We referred https://arxiv.org/pdf/2104.11178v1.pdf and
directly encode audio for transformers and didn't use Mel Spectogram or other feature extractor.

#### Results
Result with 
3 audio_representation_layers
2 text_representation_layers
2 cross_attention_layers

We get an average validation f1 of 1.
1. 'action_f1': 1.0 and 'action_accuracy' :100 %
2. 'object_f1': 1.0 and 'object_accuracy' :100 %
3. 'position_f1': 1.0 and 'position_accuracy' :100 %

check logs/train_logs.log line 933 and can also refer eval_logs.log








