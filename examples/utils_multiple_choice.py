# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import logging
import sys
from io import open
import json
import csv
import tqdm
from nltk import pos_tag, word_tokenize
from typing import List
from transformers import PreTrainedTokenizer
import random


logger = logging.getLogger(__name__)
pos_tag_map = {"CC": 1, "CD": 2, "DT": 3, "EX": 4, "FW": 5, "IN": 6, "JJ": 7, "JJR": 8, "JJS": 9, "LS": 10, "MD": 11,
               "NN": 12, "NNS": 13, "NNP": 14, "NNPS": 15, "PDT": 16, "POS": 17, "PRP": 18, "PRP$": 19, "RB": 20,
               "RBR": 21, "RBS": 22, "RP": 23, "SYM": 24, "TO": 25, "UH": 26, "VB": 27, "VBD": 28, "VBG": 29, "VBN": 30,
               "VBP": 31, "VBZ": 32, "WDT": 33, "WP": 34, "WP$": 35, "WRB": 36}


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, context, endings, label=None):
        """Constructs a InputExample.
        Args:
            example_id: Unique id for the example.
            context: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to context' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples_ori, but not for test examples_ori.
        """
        self.example_id = example_id
        self.question = question
        self.context = context
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'context_end': context_end,
                'option_start': option_start,
                'option_end': option_end,
                'pos_tags': pos_tags,
            }
            for input_ids, input_mask, segment_ids, context_end, option_start, option_end, pos_tags in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples_ori for the training and dev sets."""
        examples = []
        for suffix in ["_high.json", "_middle.json"]:
            with open(data_dir + "/" + type + suffix, "r") as f:
                data = json.load(f)
                for i in range(len(data)):
                    # entry_id, context, question, answer
                    d = [data[i]["entry_id"], data[i]["context"].lower(), data[i]["question"].lower(),
                         data[i]["answer"].lower()]
                    # choice
                    for j in range(len(data[i]["choices_list"])):
                        d += [data[i]["choices_list"][j].lower()]
                    for k in range(4):
                        if d[4 + k] == d[3]:
                            answer = str(k)
                            break
                    label = answer
                    examples.append(InputExample(example_id=data[0], context=d[1], question=d[2],
                                                 endings=[d[4], d[5], d[6], d[7]], label=label))
        return examples


class RaceHProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/" + type + "_high.json", "r") as fh:
            data = json.load(fh)
            for i in range(len(data)):
                # entry_id, context, question, answer
                d = [data[i]["entry_id"], data[i]["context"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(InputExample(example_id=data[0], context=d[1], question=d[2],
                                             endings=[d[4], d[5], d[6], d[7]], label=label))
        return examples


class RaceMProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/" + type + "_middle.json", "r") as fm:
            data = json.load(fm)
            for i in range(len(data)):
                # entry_id, context, question, answer
                d = [data[i]["entry_id"], data[i]["context"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(InputExample(example_id=data[0], context=d[1], question=d[2],
                                             endings=[d[4], d[5], d[6], d[7]], label=label))
        return examples


class DreamProcessor(DataProcessor):

    def __init__(self):
        self.data_pos = {"train": 0, "dev": 1, "test": 2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples_ori for the training and dev sets."""
        if len(self.D[self.data_pos[type]]) == 0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/" + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        # entry_id, context, question, answer
                        d = [data[i]["entry_id"], data[i]["context"].lower(), data[i]["question"].lower(),
                             data[i]["answer"].lower()]
                        # choice
                        for j in range(len(data[i]["choices_list"])):
                            d += [data[i]["choices_list"][j].lower()]
                        self.D[sid] += [d]

        data = self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                if data[i][4 + k] == data[i][3]:
                    answer = str(k)
            label = answer
            examples.append(
                InputExample(example_id=data[0], context=d[1], question=d[2], endings=[d[4], d[5], d[6]], label=label))
        return examples


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_segment_id=0,
        pad_on_left=False,
        mask_padding_with_zero=True,
        truncation_strategy="longest_first",
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples_ori to features")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        # record end positions of two parts which need interaction such as Passage and Question, for later separating them
        for ending in example.endings:
            text_a = example.context  # 每一个QA对的context
            text_b = example.question + " " + ending  # 每一个QA对
            special_tok_len = 3  # [CLS] [SEP] [SEP]
            t_q_len = len(tokenizer.tokenize(example.question))
            b_tokens = tokenizer.tokenize(text_b)
            t_b_len = len(b_tokens)
            context_max_len = max_length - special_tok_len - t_b_len
            context_tokens = tokenizer.tokenize(example.context)
            t_c_len = len(context_tokens)
            if t_c_len > context_max_len:
                t_c_len = context_max_len
                context_tokens = context_tokens[:context_max_len]
            assert (t_b_len + t_c_len <= max_length)
            # PQA都不包含特殊token
            context_end = t_c_len + 1
            option_start = t_c_len + t_q_len + 2
            option_end = t_c_len + t_b_len + 2
            pos_tags = [0] * max_length
            con_pos_tag = pos_tag(word_tokenize(text_a))
            que_pos_tag = pos_tag(word_tokenize(text_b))

            curr_index = 0
            for i in range(len(context_tokens)):
                context_tokens[i] = context_tokens[i].replace("▁", "")
                if context_tokens[i] == "":
                    pos_tags[i + 1] = 38
                else:
                    # 有些外文字符无法匹配，就默认当前的POS
                    if context_tokens[i] not in con_pos_tag[curr_index][0] and context_tokens[i] in con_pos_tag[curr_index + 1][0]:
                        curr_index += 1
                    curr_pos = con_pos_tag[curr_index][1]
                    if curr_pos in pos_tag_map:
                        pos_tags[i + 1] = pos_tag_map[curr_pos]
                    else:
                        pos_tags[i + 1] = 38

            curr_index = 0
            for i in range(t_b_len):
                b_tokens[i] = b_tokens[i].replace("▁", "")
                if b_tokens[i] == "":
                    pos_tags[i + context_end + 1] = 38
                else:
                    # 有些外文字符无法匹配，就默认当前的POS
                    if b_tokens[i] not in que_pos_tag[curr_index][0] and b_tokens[i] in que_pos_tag[curr_index + 1][0]:
                        curr_index += 1
                    curr_pos = que_pos_tag[curr_index][1]
                    if curr_pos in pos_tag_map:
                        pos_tags[i + context_end + 1] = pos_tag_map[curr_pos]
                    else:
                        pos_tags[i + context_end + 1] = 38

            # 特殊token
            pos_tags[0] = 37
            pos_tags[context_end] = 37
            pos_tags[option_end] = 37

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation_strategy=truncation_strategy
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            assert (len(input_ids[t_c_len + t_b_len:]) == special_tok_len)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            pad_token = tokenizer.pad_token_id

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append(
                (input_ids, attention_mask, token_type_ids, context_end, option_start, option_end, pos_tags))

        label = label_map[example.label]
        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label
            )
        )
    return features


processors = {
    "race": RaceProcessor,
    "racem": RaceMProcessor,
    "raceh": RaceHProcessor,
    "dream": DreamProcessor
}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race", 4,
    "racem", 4,
    "raceh", 4,
    "dream", 3,
}
