# POI-Net

Codes for ACL 2022: Lite Unified Modeling for Discriminative Reading Comprehension

The codes are written based on https://github.com/huggingface/transformers.

The codes are tested with pytorch 1.5.0 and python 3.7.

POI-Net can deal with RACE/RACE_middle/RACE_high, DREAM, SQuAD 1.1/2.0,  task whose task_name is 'race'/'racem'/'raceh', 'dream' and 'squad'.

## Run Dataset

Download the original RACE, DREAM, SQuAD 1.1 and SQuAD 2.0 online and save them at './{race_data, dream_data, squad_data/}'

## Run POI-Net

Train on DREAM: (`sh_dream.sh`)

Train on RACE: (`sh_race.sh`)

Train on SQuAD 1.1: (`sh_squad1.sh`)

Train on SQuAD 2.0: (`sh_squad2.sh`)