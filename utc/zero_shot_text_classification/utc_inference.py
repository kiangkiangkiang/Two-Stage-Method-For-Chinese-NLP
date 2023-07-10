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
from dataclasses import dataclass, field
from paddlenlp.utils.log import logger
from metric import MetricReport

import paddle
from paddle.metric import Accuracy
from sklearn.metrics import f1_score
from utils import UTCLoss, read_inference_dataset

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import UTC, AutoTokenizer, PretrainedTokenizer
from typing import Any, Dict, List, Optional


@dataclass
class DataArguments:
    test_path: str = field(default="./data/processed_data_8000/processed_data.json", metadata={"help": "Test dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})
    single_label: str = field(default=False, metadata={"help": "Predict exactly one label per sample."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="utc-base", metadata={"help": "Build-in pretrained model."})
    model_path: str = field(default=None, metadata={"help": "Build-in pretrained model."})


class InferenceUTCTemplate(UTCTemplate):

    template_special_tokens = ["text", "hard", "sep", "cls", "options"]

    def __init__(self, tokenizer: PretrainedTokenizer, max_length: int, prompt: str = None):
        prompt = (
            (
                "{'options': 'choices', 'add_omask': True, 'position': 0, 'token_type': 1}"
                "{'sep': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
            )
            if prompt is None
            else prompt
        )
        super(UTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        if not self._has_options():
            raise ValueError(
                "Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt)
            )

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(UTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token
        
        # breakpoint()
        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):
        input_dict = super(UTCTemplate, self).encode(example)

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict["omask_positions"] = (
            np.where(np.array(input_dict["input_ids"]) == omask_token_id)[0].squeeze().tolist()
        )

        sep_positions = (
            np.where(np.array(input_dict["input_ids"]) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
        )
        input_dict["cls_positions"] = sep_positions[0]

        # Limit the maximum position ids.
        position_ids = np.array(input_dict["position_ids"])
        position_ids[position_ids > self.max_position_id] = self.max_position_id
        input_dict["position_ids"] = position_ids.tolist()

        return input_dict

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict




def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = InferenceUTCTemplate(tokenizer, training_args.max_seq_length)

    # Load and preprocess dataset.
    if data_args.test_path is not None:
        test_ds = load_dataset(read_inference_dataset, data_path=data_args.test_path, lazy=False)

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )
    if model_args.model_path is not None:
        model_state = paddle.load(os.path.join(model_args.model_path, "model_state.pdparams"))
        prompt_model.set_state_dict(model_state)

    # Define the metric function.
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=UTCLoss(),
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
    )

    if data_args.test_path is not None:
        test_ret = trainer.predict(test_ds)
        with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as fp:
            if data_args.single_label:
                preds = paddle.nn.functional.softmax(paddle.to_tensor(test_ret.predictions), axis=-1)
                for index, pred in enumerate(preds):
                    result = {"id": index}
                    result["labels"] = paddle.argmax(pred).item()
                    result["probs"] = pred[result["labels"]].item()
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                preds = paddle.nn.functional.sigmoid(paddle.to_tensor(test_ret.predictions))
                for index, pred in enumerate(preds):
                    result = {"id": index}
                    result["labels"] = paddle.where(pred > data_args.threshold)[0].tolist()
                    result["probs"] = pred[pred > data_args.threshold].tolist()
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
