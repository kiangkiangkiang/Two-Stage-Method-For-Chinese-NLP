# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import json
import os

import numpy as np
import paddle

from paddlenlp.utils.log import logger
from paddlenlp import Taskflow
from typing import List
from tqdm import tqdm
from uie_model.filter_text import *


def read_local_dataset_by_chunk(data_path, data_file=None, is_test=False, max_seq_len=512, template_tokens_len=None):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """
    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]
    skip_count = 0
    max_content_len = max_seq_len - template_tokens_len
    std_keys = ["text_a", "text_b", "question", "choices", "labels"]
    verdict_id = 0

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in fp:
                example = json.loads(example.strip())
                total_chunks = 0
                example_collector = []
                while example["text_a"]:
                    this_example = example.copy()
                    this_example["text_a"] = example["text_a"][:max_content_len]
                    # logger.debug(this_example["text_a"])
                    if (
                        len(this_example["choices"]) < 2
                        or not isinstance(this_example["text_a"], str)
                        or len(this_example["text_a"]) < 3
                    ):
                        skip_count += 1
                        break
                    if "text_b" not in this_example:
                        this_example["text_b"] = ""
                    if not is_test or "labels" in this_example:
                        if not isinstance(this_example["labels"], list):
                            this_example["labels"] = [this_example["labels"]]
                        one_hots = np.zeros(len(this_example["choices"]), dtype="float32")
                        for x in this_example["labels"]:
                            one_hots[x] = 1
                        this_example["labels"] = one_hots.tolist()

                    if is_test:
                        example_collector.append(this_example)
                        # yield this_example
                        continue
                    std_keys = ["text_a", "text_b", "question", "choices", "labels"]
                    # std_example = {k: this_example[k] for k in std_keys if k in this_example}
                    example_collector.append({k: this_example[k] for k in std_keys if k in this_example})
                    # yield std_example

                    example["text_a"] = example["text_a"][max_content_len:]
                    total_chunks += 1

                logger.debug(f"{verdict_id}")
                for each_example in example_collector:
                    each_example["id"] = verdict_id
                    each_example["total_chunks"] = total_chunks
                    yield each_example

                verdict_id += 1

    logger.warning(f"Skip {skip_count} examples.")


def read_local_dataset(data_path, data_file=None, is_test=False):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """
    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]
    skip_count = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in fp:
                example = json.loads(example.strip())
                if len(example["choices"]) < 2 or not isinstance(example["text_a"], str) or len(example["text_a"]) < 3:
                    skip_count += 1
                    continue
                if "text_b" not in example:
                    example["text_b"] = ""
                if not is_test or "labels" in example:
                    if not isinstance(example["labels"], list):
                        example["labels"] = [example["labels"]]
                    one_hots = np.zeros(len(example["choices"]), dtype="float32")
                    for x in example["labels"]:
                        one_hots[x] = 1
                    example["labels"] = one_hots.tolist()

                if is_test:
                    yield example
                    continue
                std_keys = ["text_a", "text_b", "question", "choices", "labels"]
                std_example = {k: example[k] for k in std_keys if k in example}
                yield std_example
    logger.warning(f"Skip {skip_count} examples.")


def read_local_dataset_with_uie_filter(
    data_path,
    data_file=None,
    is_test=False,
    max_seq_len: int = 512,
    special_word_len: int = 200,
    uie_model_name_or_path: str = "./uie_model/model_best/",
    schema: List[str] = ["原告年齡", "肇事過失責任比例", "受有傷害"],
    dynamic_adjust_length: bool = True,
    threshold: float = 0,
):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """

    # TODO Now only for gpu
    uie = Taskflow("information_extraction", task_path=uie_model_name_or_path, schema=schema, precision="fp16")
    max_content_len = max_seq_len - special_word_len
    uie_miss_cases = 0
    miss_recoder = []

    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]
    skip_count = 0
    total_example = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in tqdm(fp):
                example = json.loads(example.strip())

                # uie filter
                uie_output = uie(example["text_a"])
                new_text = filter_text(
                    raw_text=example["text_a"],
                    uie_output=uie_output,
                    max_len_of_new_text=max_content_len,
                    threshold=threshold,
                    dynamic_adjust_length=dynamic_adjust_length,
                )
                if len(new_text) == 0 and len(example["labels"]) != 0:
                    uie_miss_cases += 1
                    example["Missing Records"] = [example["choices"][i] for i in example["labels"]]
                    miss_recoder.append(example)

                    logger.debug(f"UIE miss {uie_miss_cases}")

                example["text_a"] = new_text

                if len(example["choices"]) < 2 or not isinstance(example["text_a"], str) or len(example["text_a"]) < 3:
                    skip_count += 1
                    continue
                if "text_b" not in example:
                    example["text_b"] = ""
                if not is_test or "labels" in example:
                    if not isinstance(example["labels"], list):
                        example["labels"] = [example["labels"]]
                    one_hots = np.zeros(len(example["choices"]), dtype="float32")
                    for x in example["labels"]:
                        one_hots[x] = 1
                    example["labels"] = one_hots.tolist()

                total_example += 1

                if is_test:
                    yield example
                    continue
                std_keys = ["text_a", "text_b", "question", "choices", "labels"]
                std_example = {k: example[k] for k in std_keys if k in example}
                yield std_example
    logger.warning(f"Skip {skip_count} examples.")
    if uie_miss_cases > 0:
        logger.debug(f"Miss Cases: {miss_recoder}.")
        logger.debug(f"Number of UIE missing cases: {uie_miss_cases}. Missing rate: {uie_miss_cases/total_example}")
        breakpoint()


def get_template_tokens_len(tokenizer, label_file):
    """
    Template: [CLS] [O-MASK] label-1 [O-MASK] label-2 ... [O-MASK] label-end [SEP] contents [SEP] [SEP]

    Args:
        tokenizer (_type_): _description_
        label_file (_type_): _description_
    """
    all_labels = []
    with open(label_file, "r") as f:
        for each_label in f:
            all_labels.append("[O-MASK]")
            all_labels.append(each_label.strip())
    text = "".join(all_labels)
    prefix_text = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    return len(prefix_text) + 2  # 2 means the last two [SEP]


class UTCLoss(object):
    def __call__(self, logit, label):
        return self.forward(logit, label)

    def forward(self, logit, label):
        logit = (1.0 - 2.0 * label) * logit
        logit_neg = logit - label * 1e12
        logit_pos = logit - (1.0 - label) * 1e12
        zeros = paddle.zeros_like(logit[..., :1])
        logit_neg = paddle.concat([logit_neg, zeros], axis=-1)
        logit_pos = paddle.concat([logit_pos, zeros], axis=-1)
        label = paddle.concat([label, zeros], axis=-1)
        logit_neg[label == -100] = -1e12
        logit_pos[label == -100] = -1e12
        neg_loss = paddle.logsumexp(logit_neg, axis=-1)
        pos_loss = paddle.logsumexp(logit_pos, axis=-1)
        loss = (neg_loss + pos_loss).mean()
        return loss


def uie_preprocessing():
    pass


def read_inference_dataset(data_path, data_file=None, options="./data/label.txt"):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """
    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]

    with open(options, "r", encoding="utf-8") as fp:
        choices = [x.strip() for x in fp]

    skip_count = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in fp:
                example = json.loads(example.strip())
                example["text_b"] = ""
                example["choices"] = choices
                yield example

    logger.warning(f"Skip {skip_count} examples.")