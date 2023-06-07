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
from paddlenlp.transformers import ErnieForMaskedLM, PretrainedTokenizer, register_base_model
from paddlenlp.transformers import UTC, ErniePretrainedModel, ErnieConfig, ErnieModel
from paddlenlp.transformers.model_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from paddlenlp.prompt.prompt_utils import signature, PromptDataCollatorWithPadding
from paddlenlp.prompt.template import PrefixTemplate, Template
from paddlenlp.prompt.verbalizer import Verbalizer
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle import Tensor, nn
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


@register_base_model
class myErnieModel(ErnieModel):
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        task_type_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                We use whole-word-mask in ERNIE, so the whole word will have the same value. For example, "使用" as a word,
                "使" and "用" will have the same value.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieModel, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieModel.from_pretrained('ernie-1.0')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        use_cache = use_cache if use_cache is not None else False
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        embedding_output = paddle.to_tensor([0.0])
        counter = 0

        for i in range(len(input_ids)):
            counter += 1
            if attention_mask[i] is None:
                attention_mask[i] = paddle.unsqueeze(
                    (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
                )
                if past_key_values is not None:
                    batch_size = past_key_values[0][0].shape[0]
                    past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                    attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)

            # For 2D attention_mask from tokenizer
            elif attention_mask[i].ndim == 2:
                attention_mask[i] = paddle.unsqueeze(attention_mask[i], axis=[1, 2]).astype(paddle.get_default_dtype())
                attention_mask[i] = (1.0 - attention_mask[i]) * -1e4

            attention_mask[i].stop_gradient = True

            embedding_output += paddle.mean(
                self.embeddings(
                    input_ids=input_ids[i],
                    position_ids=position_ids[i],
                    token_type_ids=token_type_ids[i],
                    task_type_ids=task_type_ids,
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_key_values_length,
                ),
                0,
                keepdim=True,
            )

        embedding_output = embedding_output / counter
        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask[0][0],
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )


class myUTC(ErniePretrainedModel):
    """
    Ernie Model with two linear layer on the top of the hidden-states output to compute
    probability of candidate labels, designed for Unified Tag Classification.
    """

    def __init__(self, config: ErnieConfig):
        super(myUTC, self).__init__(config)
        self.ernie = myErnieModel(config)
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(mean=0.0, std=config.initializer_range)
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, weight_attr=weight_attr
        )

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
        for i in range(len(input_ids)):
            input_ids[i]

        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        omask_positions = omask_positions[0][0]
        cls_positions = cls_positions[0][0]

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
        breakpoint()
        return_dict = return_dict if return_dict is not None else False
        return_hidden_states = kwargs.get("return_hidden_states", False)
        breakpoint()
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "masked_positions": masked_positions,
            "soft_token_ids": soft_token_ids,
            "attention_mask": attention_mask,
            "encoder_ids": encoder_ids,
            **kwargs,
        }
        breakpoint()
        input_dict = self.template.process_batch(input_dict)
        breakpoint()
        input_dict = {**input_dict, **kwargs}
        model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
        if "masked_positions" in model_inputs:
            model_inputs.pop("masked_positions")
        model_outputs = self.plm(**model_inputs, return_dict=True)
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
