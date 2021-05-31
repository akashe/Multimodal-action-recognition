#Multimodal action recognition:

#### Task Definition


#### Set up virtual environment

python -m spacy download en

#### Model Variants

#####v1: Full Transformer 
Referencing https://arxiv.org/pdf/2104.11178v1.pdf you can
directly encode audio for transformers and not use Mel Spectogram

Experiment 1:
Result with 
2 audio_representation_layers
1 text_representation_layers
1 cross_attention_layers

Epoch: 50 | Time: 0m 23s
Train Loss: 0.020
Val. Loss: 0.001
Train f1: {'action_f1': 0.8819732570377935, 'object_f1': 0.9919541110008183, 'position_f1': 0.9953303464356397} 
 Valid f1: {'action_f1': 0.8747055626826312, 'object_f1': 0.17529257716746305, 'position_f1': 1.0}
Train action accuracy: 83.120 	 Valid action accuracy: 81.960
Train object accuracy: 99.198 	 Valid object accuracy: 17.809
Train location accuracy: 99.534 	 Valid location accuracy: 100.000

tensorboard file v1/base

Experiment 2: 
Weighted mean based on vocab sizes.

Experiment 3:
Seperate heads for all three
#####v2: Using pre-trained audio embedding 


#### Train script


#### Test script


#### Tensorboard


