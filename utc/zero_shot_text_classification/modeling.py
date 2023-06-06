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


from typing import Any, Dict, Optional
import numpy as np
import paddle
from paddle.static import InputSpec
from paddlenlp.transformers import ErnieForMaskedLM, PretrainedTokenizer
from paddlenlp.transformers import UTC, ErniePretrainedModel, ErnieConfig, ErnieModel
from paddlenlp.transformers.model_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from paddlenlp.prompt.prompt_utils import signature, PromptDataCollatorWithPadding
from paddlenlp.prompt.template import PrefixTemplate, Template
from paddlenlp.prompt.verbalizer import Verbalizer
from dataclasses import dataclass
from typing import List, Dict, Any
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle import Tensor
from paddlenlp.prompt import UTCTemplate

# ''.join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, :]))
# len(''.join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, :])))

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from paddlenlp.prompt.prompt_tokenizer import MLMPromptTokenizer

from paddlenlp.utils.log import logger


class myMLMPromptTokenizer(MLMPromptTokenizer):

    omask_token = "[O-MASK]"

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, inputs: List[Dict[str, Any]]):
        part_do_truncate = [part["do_truncate"] for part in inputs]
        encoded_inputs = defaultdict(list)
        option_length = None
        last_position = 1  # Id 0 denotes special token '[CLS]'.
        last_token_type = 0
        orig_input_ids = []
        for index, part in enumerate(inputs):
            # Create input_ids.
            soft_token_ids = part.get("soft_tokens", None)
            if soft_token_ids is None or len(soft_token_ids) == 1 and soft_token_ids[0] == 0:
                breakpoint()
                orig_input_ids.append(
                    self.tokenizer.encode(part["text"], add_special_tokens=False, return_token_type_ids=False)[
                        "input_ids"
                    ]
                )
                breakpoint()
            else:
                orig_input_ids.append(soft_token_ids)
        max_lengths = self._create_max_lengths_from_do_truncate(orig_input_ids, part_do_truncate)

        for index, part in enumerate(inputs):
            # Create input_ids.
            soft_token_ids = part.get("soft_tokens", None)
            if soft_token_ids is None or len(soft_token_ids) == 1 and soft_token_ids[0] == 0:
                if self.tokenizer.truncation_side == "left":
                    input_ids = orig_input_ids[index][-max_lengths[index] :]
                else:
                    input_ids = orig_input_ids[index][: max_lengths[index]]
                encoded_inputs["soft_token_ids"].append([0] * len(input_ids))
            else:
                input_ids = soft_token_ids
                encoded_inputs["soft_token_ids"].append(soft_token_ids)
            encoded_inputs["input_ids"].append(input_ids)
            part_length = len(input_ids)

            # Create position_ids.
            position_ids, last_position = self._create_position_ids_from_part(input_ids, part, last_position)
            encoded_inputs["position_ids"].append(position_ids)

            # Create token_type_ids.
            if "token_types" in part:
                last_token_type = part["token_types"]
            encoded_inputs["token_type_ids"].append([last_token_type] * part_length)

            # Create other features like encoder_ids.
            for name in part:
                if name not in ["text", "soft_tokens", "positions", "token_types"]:
                    encoded_inputs[name].append([part[name]] * part_length)

            # Record the length of options if exists.
            if self.omask_token in part["text"]:
                option_length = len(input_ids)

        encoded_inputs.pop("do_truncate")
        encoded_inputs = self.join(encoded_inputs)
        encoded_inputs = self.add_special_tokens(encoded_inputs)
        attention_mask = self._create_attention_mask(encoded_inputs["input_ids"], option_length)
        if attention_mask is not None:
            encoded_inputs["attention_mask"] = attention_mask
        masked_positions = self._create_masked_positions(encoded_inputs["input_ids"], encoded_inputs["soft_token_ids"])
        if masked_positions is not None:
            encoded_inputs["masked_positions"] = masked_positions
        breakpoint()
        return encoded_inputs


class myUTCTemplate(Template):
    """
    Template for Unified Tag Classification.
    """

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
        super(myUTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        self.prompt_tokenizer = myMLMPromptTokenizer(tokenizer, max_length)
        if not self._has_options():
            raise ValueError("Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt))

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(myUTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token
        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):

        input_text = self.build_inputs_with_prompt(example)
        input_names, input_values = ["text"], [input_text]
        for name in self.input_feature_names:
            input_names.append(name)
            input_values.append(getattr(self, name, None))

        inputs = []
        for value in list(zip(*input_values)):
            inputs.append(dict(zip(input_names, value)))

        input_dict = self.prompt_tokenizer(inputs)

        unused_example = {k: v for k, v in example.items() if k not in self.example_keys}
        input_dict = {**input_dict, **unused_example}

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict["omask_positions"] = (
            np.where(np.array(input_dict["input_ids"]) == omask_token_id)[0].squeeze().tolist()
        )

        sep_positions = np.where(np.array(input_dict["input_ids"]) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
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


@dataclass
class myDataCollator(PromptDataCollatorWithPadding):
    default_model_input_names: List = (
        "input_ids",
        "token_type_ids",
        "special_tokens_mask",
        "offset_mapping",
        "position_ids",
        "id",
        "total_chunks",
    )


class myUTC(ErniePretrainedModel):
    """
    Ernie Model with two linear layer on the top of the hidden-states output to compute
    probability of candidate labels, designed for Unified Tag Classification.
    """

    def __init__(self, config: ErnieConfig):
        super(myUTC, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.predict_size = 64
        self.linear_q = paddle.nn.Linear(config.hidden_size, self.predict_size)
        self.linear_k = paddle.nn.Linear(config.hidden_size, self.predict_size)

    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        omask_positions,
        cls_positions,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor):
                See :class:`ErnieModel`.
            position_ids (Tensor):
                See :class:`ErnieModel`.
            attention_mask (Tensor):
                See :class:`ErnieModel`.
            omask_positions (Tensor of shape `(batch_size, max_option)`):
                Masked positions of [O-MASK] tokens padded with 0.
            cls_positions (Tensor of shape `(batch_size)`):
                Masked positions of the second [CLS] token.
            labels (Tensor of shape `(num_labels_in_batch,)`, optional):
                Labels for computing classification loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        breakpoint()
        sequence_output = paddle.to_tensor([0.0])
        for i in range(len(input_ids)):
            outputs = self.ernie(
                input_ids[i],
                token_type_ids=token_type_ids[i],
                position_ids=position_ids[i],
                attention_mask=attention_mask[i],
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            breakpoint()
            sequence_output += paddle.sum(outputs[0], 0, keepdim=True)
            breakpoint()
            # if i == len(input_ids) - 1:
        breakpoint()

        # else:
        # sequence_output = outputs[0]

        batch_size, seq_len, hidden_size = sequence_output.shape
        flat_sequence_output = paddle.reshape(sequence_output, [-1, hidden_size])
        flat_length = paddle.arange(batch_size) * seq_len
        flat_length = flat_length.unsqueeze(axis=1).astype("int64")

        cls_output = paddle.tensor.gather(flat_sequence_output, cls_positions + flat_length.squeeze(1))
        q = self.linear_q(cls_output)

        option_output = paddle.tensor.gather(flat_sequence_output, paddle.reshape(omask_positions + flat_length, [-1]))
        option_output = paddle.reshape(option_output, [batch_size, -1, hidden_size])
        k = self.linear_k(option_output)

        option_logits = paddle.matmul(q.unsqueeze(1), k, transpose_y=True).squeeze(1)
        option_logits = option_logits / self.predict_size**0.5
        for index, logit in enumerate(option_logits):
            option_logits[index] -= (1 - (omask_positions[index] > 0).astype("float32")) * 1e12

        loss = None
        if not return_dict:
            output = (option_logits,)
            if output_hidden_states:
                output = output + (outputs.hidden_states,)
            if output_attentions:
                output = output + (output.attentions,)
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=option_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PromptModelForSequenceClassification(paddle.nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        template: Template,
        verbalizer: Optional[Verbalizer] = None,
        freeze_plm: bool = False,
        freeze_dropout: bool = False,
    ):
        super(PromptModelForSequenceClassification, self).__init__()
        self.plm = model
        self.sigmoid = F.sigmoid
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
            if self.freeze_dropout:
                self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id
        if isinstance(self.template, PrefixTemplate):
            self.plm = self.template.process_model(self.plm)
            self.forward_keys.append("past_key_values")

    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        masked_positions: Optional[paddle.Tensor] = None,
        soft_token_ids: Optional[paddle.Tensor] = None,
        encoder_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ):
        return_dict = return_dict if return_dict is not None else False
        return_hidden_states = kwargs.get("return_hidden_states", False)
        this_kwargs = {}
        model_outputs = paddle.to_tensor([0.0])
        for i in range(len(input_ids)):
            for k in kwargs:
                this_kwargs[k] = kwargs[k][i]
            input_dict = {
                "input_ids": input_ids[i],
                "token_type_ids": token_type_ids[i],
                "position_ids": position_ids[i],
                "masked_positions": masked_positions,
                "soft_token_ids": soft_token_ids[i],
                "attention_mask": attention_mask[i],
                "encoder_ids": encoder_ids,
                **this_kwargs,
            }
            input_dict = self.template.process_batch(input_dict)
            input_dict = {**input_dict, **this_kwargs}
            model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
            if "masked_positions" in model_inputs:
                model_inputs.pop("masked_positions")

            # model_outputs = self.plm(**model_inputs, return_dict=True)

            if i == len(input_ids) - 1:
                # model_outputs += paddle.mean(self.plm(**model_inputs, return_dict=True).logits, 0, keepdim=True)
                model_outputs += paddle.mean(self.plm(**model_inputs, return_dict=True).logits, 0, keepdim=True)

            else:
                model_outputs += paddle.mean(
                    self.plm(**model_inputs, return_dict=True, output_hidden_states=True).logits.detach(),
                    0,
                    keepdim=True,
                )

        # logger.debug(f"logits: {model_outputs / len(input_ids)}")
        # logger.debug(f"logits: {self.sigmoid(model_outputs / len(input_ids))}")
        model_outputs = MultipleChoiceModelOutput(
            loss=None, logits=model_outputs / len(input_ids), hidden_states=None, attentions=None
        )

        if isinstance(model_outputs, MaskedLMOutput):
            if self.verbalizer is not None:
                logits = self.verbalizer.process_outputs(model_outputs.logits, input_dict["masked_positions"])
                num_labels = len(self.verbalizer.label_words)
            else:
                raise Exception("Verbalizer is required when model uses the MaskedLM head")
        elif isinstance(model_outputs, SequenceClassifierOutput):
            logits = model_outputs.logits
            num_labels = self.plm.num_labels if self.plm.num_labels is not None else self.plm.num_labels
        elif isinstance(model_outputs, MultipleChoiceModelOutput):
            logits = model_outputs.logits
            num_labels = -1
        else:
            raise Exception(f"Model type not support yet: {type(model_outputs)}")

        loss = None
        if labels is not None:
            if num_labels == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif num_labels > 0 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, num_labels)), labels.reshape((-1,)))
            else:
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            if return_hidden_states:
                output = output + (model_outputs.logits,)
            if loss is not None:
                return (loss,) + output
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.logits,
        )

    def prompt_parameters(self):
        """
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer is not None:
            params += [p for p in self.verbalizer.parameters()]
        return params

    def get_input_spec(self):
        template_keywords = self.template.extract_template_keywords(self.template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
        ]
        if "mask" in template_keywords:
            input_spec.append(InputSpec(shape=[None], dtype="int64", name="masked_positions"))
        if "soft" in template_keywords:
            # Add placeholder for argument `masked_positions` if not exists.
            if "mask" not in template_keywords:
                input_spec.append(None)
            input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="soft_token_ids"))
            if "encoder" in template_keywords:
                input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="encoder_ids"))
        return input_spec
