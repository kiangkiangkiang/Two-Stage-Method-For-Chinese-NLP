from paddlenlp.prompt import PromptTrainer, PrefixModelForCausalLM
from typing import Optional
from paddle.io import DistributedBatchSampler
from paddlenlp.trainer.trainer_utils import has_length
import numpy as np
import pandas as pd
import math
import paddle.distributed as dist
from paddle.distributed import fleet
from paddlenlp.layers.lora import LoRAModel
from paddle.io import DataLoader
from paddlenlp.utils.log import logger
from paddlenlp.utils.env import LORA_WEIGHT_FILE_NAME, PREFIX_WEIGHT_FILE_NAME, PADDLE_WEIGHT_FILE_NAME
from paddlenlp.trainer.utils.helper import nested_concat, nested_numpify, nested_truncate, nested_detach
from paddlenlp.data import DataCollator
from paddlenlp.utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from paddlenlp.trainer.trainer_utils import (
    EvalPrediction,
    EvalLoopOutput,
    has_length,
    find_batch_size,
    IterableDatasetShard,
    get_last_checkpoint,
    ShardingOption,
    speed_metrics,
    TrainOutput,
)
import paddle.nn.functional as F
from paddle import nn
from tqdm.auto import tqdm
import os
import time
import sys
import paddle
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.trainer.trainer_callback import TrainerState
from typing import Union, Optional, Dict, Tuple, List, Any
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)

TRAINER_STATE_NAME = "trainer_state.json"

import inspect


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


from collections import defaultdict


class myUTCTrainer(PromptTrainer):
    logits_collector = {}
    inputs_collector = defaultdict(dict)
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

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            if not os.path.isfile(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix))
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            # TODO: Need to load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)),
                return_numpy=True,
            )
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)

            # release memory
            del state_dict

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = len(self.train_dataset)

            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = int(len(self.train_dataset) * args.num_train_epochs)

            if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
                if max_steps // args.eval_steps < args.minimum_eval_times:
                    exp_step = max_steps / args.minimum_eval_times
                    exp_step = max(int(exp_step - exp_step % 10), 10)
                    logger.info("Reset eval step by minimum_eval_times to %d" % exp_step)
                    args.eval_steps = exp_step
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Total num train samples = {num_train_samples}")
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.info(f"  Number of trainable parameters = {per_device_trainable_numel} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
            if parts_num > 1:
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype="int64")
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = trainable_numel_tensor.item() // self.args.dataset_world_size
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.info(f"  Number of trainable parameters = {trainable_numel} (all devices, roughly)")

        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                    logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from paddlenlp.trainer.plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step = -1

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                dp_enabled = (
                    self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
                )
                forbidden_no_sync = False
                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and  no_sync API
                if self.sharding and (ShardingOption.SHARD_OP not in self.args.sharding):
                    forbidden_no_sync = True
                # hybrid_parallel (tp or pp) should not no_sync
                if self.args.use_hybrid_parallel and (
                    self.args.tensor_parallel_degree > 1 or self.args.pipeline_parallel_degree > 1
                ):
                    forbidden_no_sync = True

                availiable_no_sync = dp_enabled and not forbidden_no_sync

                is_no_sync = (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and availiable_no_sync
                    and args._no_sync_in_gradient_accumulation
                ) or (args.recompute and availiable_no_sync)
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if isinstance(tr_loss_step, int):
                    continue
                tr_loss += tr_loss_step

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Maunally collect gradients when group_sharded_parallel can't accept dp_group
                    # Case 1: Use sharding stage 2/3 with dp
                    # Case 2: Use recompute and dp
                    # local_rank != -1 don't means dp in networks.
                    if self.sharding and ShardingOption.SHARD_OP not in self.args.sharding:
                        if self.args.data_parallel_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                            fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())
                            if ShardingOption.FULL_SHARD in self.args.sharding:
                                # Why need sync on parm again ?
                                # TODO: fix this.
                                for p in model.parameters():
                                    if hasattr(p, "bw_storage"):
                                        assert p.grad is None, "This case shouldn't happen."
                                        p.bw_storage.scale_(1.0 / self.dp_group.nranks)
                                        paddle.distributed.all_reduce(p.bw_storage, group=self.dp_group)

                    # Case 2: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    elif args.recompute and availiable_no_sync:
                        fused_allreduce_gradients(list(model.parameters()), None)

                    # pipeline parallel mode,  handle gradient merge here
                    if args.pipeline_parallel_degree > 1:
                        real_accumulate_steps = args.gradient_accumulation_steps
                        if hasattr(model, "_delay_scale_loss"):
                            if model._delay_scale_loss:
                                real_accumulate_steps *= model.accumulate_steps

                        for p in model._layers.parameters():
                            if hasattr(p, "main_grad") and p.main_grad is not None:
                                assert p.grad is None
                                p.main_grad = p.main_grad.scale(1.0 / real_accumulate_steps)
                            elif p.grad is not None:
                                p.grad = p.grad.scale(1.0 / real_accumulate_steps)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler._scale.numpy()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale.numpy()
                        optimizer_was_run = not self.scaler._cache_founf_inf
                        if not optimizer_was_run:
                            logger.warning(
                                f"optimizer not run, scale_before: {scale_before[0]}, scale_after: {scale_after[0]}"
                            )
                    else:
                        self.optimizer.step()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    self.optimizer.clear_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            best_model_path = os.path.join(
                self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path, return_numpy=True)
                # If the model is on the GPU, it still works!
                self._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

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
                    loss, outputs, labels = self.compute_loss(model, inputs, return_outputs=True)

                if not isinstance(loss, int):
                    loss = loss.mean().detach()
                    outputs = outputs.detach()
                    labels = labels.detach()
                else:
                    return (loss, outputs, labels)

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs
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

    def compute_loss_tmp(self, model, inputs, return_outputs=False):
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
        # breakpoint()
        loss = 0
        logits = 0
        for verdict_group in pd.unique(inputs["id"]):
            labels_output = paddle.unsqueeze(labels[inputs["id"] == verdict_group][0], 0)

            total_num_of_verdict = inputs["total_chunks"][inputs["id"] == verdict_group][0].item()

            # with paddle.no_grad():
            # self.logits_collector[verdict_group] += outputs[inputs["id"] == verdict_group].sum(axis=0, keepdim=True)

            # self.logits_collector[verdict_group] = paddle.unsqueeze(
            #    paddle.sum(outputs[inputs["id"] == verdict_group], 0), 0
            # )
            # breakpoint()
            self.accumulate_verdict[verdict_group] += int(paddle.sum(inputs["id"] == verdict_group))
            # self.accumulate_verdict[verdict_group] = paddle.sum(inputs["id"] == verdict_group).item()
            # if verdict_group != current_verdict_id:
            if self.accumulate_verdict[verdict_group] == int(total_num_of_verdict):
                self.logits_collector[verdict_group] += outputs[inputs["id"] == verdict_group].sum(axis=0, keepdim=True)

                # logger.debug(self.logits_collector.keys())
                # self.logits_collector[verdict_group].stop_gradient = False
                # breakpoint()

                # breakpoint()
                logits = self.logits_collector[verdict_group] / int(total_num_of_verdict)
                loss += self.criterion(logits, labels)
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

        outputs = (loss, logits, labels_output)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, logits, labels_output) if return_outputs else loss

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
        do_model = False
        input_key = None
        loss = 0
        for verdict_group in pd.unique(inputs["id"]):
            if self.accumulate_verdict.get(verdict_group) is None:
                # self.logits_collector[k] = paddle.to_tensor(0.0)
                # self.logits_collector[k] = np.zeros((1, 46)) # 46 classes
                self.accumulate_verdict[verdict_group] = int(paddle.sum(inputs["id"] == verdict_group))
                for key in inputs:
                    self.inputs_collector[verdict_group][key] = []
                    self.inputs_collector[verdict_group][key].append(inputs[key][inputs["id"] == verdict_group])
            else:
                self.accumulate_verdict[verdict_group] += int(paddle.sum(inputs["id"] == verdict_group))
                for key in inputs:
                    self.inputs_collector[verdict_group][key].append(inputs[key][inputs["id"] == verdict_group])
            if (
                self.accumulate_verdict[verdict_group]
                == inputs["total_chunks"][inputs["id"] == verdict_group][0].item()
            ):
                do_model = True
                input_key = verdict_group

                for input_type in ["input_ids", "position_ids", "token_type_ids", "soft_token_ids"]:
                    for i in range(len(self.inputs_collector[input_key][input_type])):
                        x, y = self.inputs_collector[input_key][input_type][i].shape
                        if y == self.args.max_seq_length:
                            continue
                        y = self.args.max_seq_length - y
                        padding = paddle.to_tensor([0] * x * y).reshape((x, y))
                        self.inputs_collector[input_key][input_type][i] = paddle.concat(
                            [self.inputs_collector[input_key][input_type][i], padding], 1
                        )

                x, _, y, tmp = self.inputs_collector[input_key]["attention_mask"][0].shape
                if y != self.args.max_seq_length:
                    y = self.args.max_seq_length - y
                    padding = paddle.to_tensor([-10000] * x * y * tmp, dtype="float64").reshape((x, 1, y, tmp))
                    self.inputs_collector[input_key]["attention_mask"][0] = paddle.concat(
                        [self.inputs_collector[input_key]["attention_mask"][0], padding], 2
                    )
                    padding = paddle.to_tensor([-10000] * x * self.args.max_seq_length * y, dtype="float64").reshape(
                        (x, 1, self.args.max_seq_length, y)
                    )
                    self.inputs_collector[input_key]["attention_mask"][0] = paddle.concat(
                        [self.inputs_collector[input_key]["attention_mask"][0], padding], 3
                    )
                logits = model(**self.inputs_collector[input_key])
                # logger.debug(self.inputs_collector.keys())
                del self.accumulate_verdict[input_key]
                del self.inputs_collector[input_key]
                labels_output = paddle.unsqueeze(labels[inputs["id"] == input_key][0], 0).cpu()
                tmp = self.criterion(logits, labels_output)

                title = "Training" if model.training else "Evaluation"
                logger.debug(
                    f"Loss of Verdict ID = {input_key}. Total Chunk in the Verdict = {inputs['total_chunks'][inputs['id'] == input_key][0].item()}"
                )
                logger.debug(f"{title} Loss {tmp.item()}.")

                loss += tmp
                del tmp

        if not do_model:
            return (0, 0, 0) if return_outputs else 0

        outputs = (loss, logits, labels_output)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, logits, labels_output) if return_outputs else loss

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
                del logits
                del labels
                continue

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
