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
from test_modeling import *
import os
from collections import defaultdict
from dataclasses import dataclass, field

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from metric import MetricReport
from utils import load_local_dataset
from paddlenlp.transformers import ErnieForMaskedLM
from paddlenlp.prompt import (
    AutoTemplate,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    SoftVerbalizer,
)
from paddlenlp.trainer import EarlyStoppingCallback, PdArgumentParser
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger

from paddlenlp.trainer.trainer import *
from paddlenlp.transformers.model_utils import PretrainedModel, _add_variant, unwrap_model
from paddlenlp.prompt.prompt_utils import *

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from paddlenlp.datasets import MapDataset
import pandas as pd


def my_foreward(model, **inputs):
    logits = model(**inputs)
    return logits


class LogitsStorager(object):
    def __init__(self) -> None:
        pass


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data", metadata={"help": "The dataset dictionary includes train.txt, dev.txt and label.txt files."})
    prompt: str = field(default=None, metadata={"help": "The input prompt for tuning."})
    # dataloader_drop_last: bool = field(default=True, metadata={"help": "Drop the last for uncomplete batch"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "The build-in pretrained model or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
# yapf: enable


@dataclass
class PromptDataCollatorWithPadding(PromptDataCollatorWithPadding):
    default_model_input_names: List = (
        "input_ids",
        "token_type_ids",
        "special_tokens_mask",
        "offset_mapping",
        "position_ids",
        "id",
        "nth_chunk",
        "total_chunk",
    )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch = {}
        for key in features[0]:
            if key in self.default_model_input_names:
                batch[key] = [b[key] for b in features]

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
        )

        max_length = batch["input_ids"].shape[1]
        for key in features[0]:
            if key not in self.default_model_input_names:
                values = [b[key] for b in features if key in b]
                if len(values) < len(features):
                    continue
                if key == "masked_positions":
                    new_values = []
                    for index, value in enumerate(values):
                        value = np.array(value) + index * max_length
                        new_values.extend(value.tolist())
                    values = new_values
                elif key == "attention_mask":
                    new_values = np.ones([len(values), 1, max_length, max_length]) * -1e4
                    for index, value in enumerate(values):
                        length = len(value)
                        new_values[index][0, :length, :length] = value
                    values = new_values
                elif key in ("soft_token_ids", "encoder_ids"):
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_length - len(value))
                elif key in ("omask_positions"):
                    max_num_option = max([len(x) for x in values])
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_num_option - len(value))
                elif key == "labels":
                    if isinstance(values[0], list):
                        max_num_label = max([len(x) for x in values])
                        for index, value in enumerate(values):
                            values[index] = value + [-100] * (max_num_label - len(value))
                elif key != "cls_positions":
                    continue
                batch[key] = self._convert_to_tensors(values)

        return batch


class MyPromptTrainer(PromptTrainer):
    logits_collector = {}
    accumulate_verdict = {}

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=False,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`paddle.io.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """
        self.args.dataloader_drop_last = False
        test_dataset = self._map_dataset(test_dataset)

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="eval")

        if self._is_iterable_dataset(test_dataset):
            if self.args.dataset_world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size * self.world_size,
                collate_fn=self.data_collator,  # _get_collator_with_removed_columns
                num_workers=self.args.dataloader_num_workers,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if isinstance(loss, int):
            return loss
        else:
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None
        outputs = model(**inputs)

        for k in pd.unique(inputs["id"]):
            if self.logits_collector.get(k) is None:
                self.logits_collector[k] = paddle.to_tensor(0.0)
                # self.logits_collector[k] = np.zeros((1, 46)) # 46 classes
                self.accumulate_verdict[k] = 0
        breakpoint()
        loss = 0
        for verdict_group in pd.unique(inputs["id"]):

            total_num_of_verdict = inputs["total_chunk"][inputs["id"] == verdict_group][0].item()

            # with paddle.no_grad():
            # self.logits_collector[verdict_group] += outputs[inputs["id"] == verdict_group].sum(axis=0, keepdim=True)

            # self.logits_collector[verdict_group] = paddle.unsqueeze(
            #    paddle.sum(outputs[inputs["id"] == verdict_group], 0), 0
            # )
            self.accumulate_verdict[verdict_group] += int(paddle.sum(inputs["id"] == verdict_group))
            # self.accumulate_verdict[verdict_group] = paddle.sum(inputs["id"] == verdict_group).item()
            if self.accumulate_verdict[verdict_group] == int(total_num_of_verdict):
                self.logits_collector[verdict_group] += outputs[inputs["id"] == verdict_group].sum(axis=0, keepdim=True)
                # logger.debug(self.logits_collector.keys())
                # self.logits_collector[verdict_group].stop_gradient = False
                # breakpoint()

                # breakpoint()
                loss += self.criterion(
                    self.logits_collector[verdict_group] / int(total_num_of_verdict),
                    paddle.unsqueeze(labels[inputs["id"] == verdict_group][0], 0),
                )
                title = "Training" if model.training else "Evaluation"
                logger.debug(
                    f"Loss of Verdict ID = {verdict_group}. Total Chunk in the Verdict = {total_num_of_verdict}"
                )
                logger.debug(f"{title} Loss {loss.item()}.")
                # breakpoint()

                del self.logits_collector[verdict_group]
                del self.accumulate_verdict[verdict_group]
            else:
                self.logits_collector[verdict_group] += (
                    outputs[inputs["id"] == verdict_group].sum(axis=0, keepdim=True).detach()
                )

        outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = self.model_wrapped
        else:
            model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.dataset_world_size * max_eval_iters
            if isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.dataset_world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if isinstance(loss, int):
                loss = None

            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)

                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
            return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                if not isinstance(loss, int):
                    loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if isinstance(logits, (list, tuple)) and len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    # paddle.set_device("cpu")

    # Load the pretrained language model.
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # text = tokenizer.convert_ids_to_tokens([1, 17416, 19509, 1397, 19574, 31, 58, 72, 245, 119, 104, 19676, 505, 1079, 19619, 3930, 17, 130, 1397, 19676, 436, 131, 4552, 9490, 19505, 250, 612, 338, 2763, 12456, 171, 612, 17555, 19660, 992, 204, 19748, 20011, 140, 38, 8, 19588, 826, 3586, 28, 517, 250, 612, 196, 171, 612, 19479, 603, 19719, 755, 487, 259, 4, 160, 200, 1342, 104, 912, 19578, 119, 104, 19748, 20011, 19556, 323, 1420, 19587, 40, 19465, 15012, 755, 19977, 19927, 12052, 276, 124, 12053, 104, 259, 4, 19480, 89, 245, 1342, 104, 911, 1405, 91, 728, 798, 152, 19472, 4, 89, 245, 1789, 119, 19466, 3930, 17, 768, 136, 1900, 139, 545, 19782, 19951, 19561, 19680, 19538, 4, 19469, 1056, 19564, 41, 392, 718, 5, 41, 503, 9, 3, 2])
    # print(text, len(text), text2, text3, text4)

    # Define the template for preprocess and the verbalizer for postprocess.
    template = AutoTemplate.create_from(
        data_args.prompt,
        # prompt="{'text': 'text_a'},{'hard': '這句話要包含的要素有'},{'mask': None, 'length': 1}",
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length,
        model=model,
    )
    logger.info("Using template: {}".format(template.prompt))

    label_file = os.path.join(data_args.data_dir, "label.txt")
    with open(label_file, "r", encoding="utf-8") as fp:
        label_words = defaultdict(list)
        for line in fp:
            data = line.strip().split("==")
            word = data[1] if len(data) > 1 else data[0].split("##")[-1]
            label_words[data[0]].append(word)

    verbalizer = SoftVerbalizer(label_words, tokenizer, model)

    # Load the few-shot datasets.
    train_ds, dev_ds, test_ds = load_local_dataset(
        data_path=data_args.data_dir,
        splits=["train", "dev", "test"],
        label_list=verbalizer.labels_to_ids,
        chunk_len=training_args.max_seq_length,
        overlap_length=20,
        prompt=data_args.prompt,
        other_tokens_length=3,
    )

    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout,
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = F.sigmoid(paddle.to_tensor(eval_preds.predictions))
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
        micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
        return {
            "micro_f1_score": micro_f1_score,
            "macro_f1_score": macro_f1_score,
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
        }

    # Deine the early-stopping callback.
    callbacks = [EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.0)]

    # Initialize the trainer.
    trainer = MyPromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=PromptDataCollatorWithPadding(tokenizer, padding=True, return_tensors="pd"),
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Training.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Prediction.
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    # Export static model.
    if training_args.do_export:
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, export_type=model_args.export_type)


if __name__ == "__main__":
    main()
