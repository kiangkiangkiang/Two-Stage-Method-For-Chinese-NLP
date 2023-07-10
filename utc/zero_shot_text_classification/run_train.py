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

from dataclasses import dataclass, field
from metric import MetricReport

from paddlenlp.prompt import PromptTrainer

import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.static import InputSpec
from sklearn.metrics import f1_score
from utils import (
    UTCLoss,
    read_local_dataset,
    get_template_tokens_len,
    read_local_dataset_by_chunk,
    uie_preprocessing,
)

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)

from paddlenlp.prompt import PromptModelForSequenceClassification

from modeling import myDataCollator
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoTokenizer, export_model, UTC
import os
from paddlenlp.utils.log import logger
import numpy as np


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="./data",
        metadata={"help": "Local dataset directory including train.txt, dev.txt and label.txt (optional)."},
    )
    train_file: str = field(default="train.txt", metadata={"help": "Train dataset file name."})
    dev_file: str = field(default="dev.txt", metadata={"help": "Dev dataset file name."})
    test_file: str = field(default="test.txt", metadata={"help": "Test dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})
    single_label: str = field(default=False, metadata={"help": "Predict exactly one label per sample."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="utc-base",
        metadata={
            "help": "The build-in pretrained UTC model name or path to its checkpoints, such as "
            "`utc-xbase`, `utc-base`, `utc-medium`, `utc-mini`, `utc-micro`, `utc-nano` and `utc-pico`."
        },
    )
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    export_model_dir: str = field(default="checkpoints/model_best", metadata={"help": "The export model path."})


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
    template = UTCTemplate(tokenizer, training_args.max_seq_length)

    # template_tokens_len = get_template_tokens_len(tokenizer, os.path.join(data_args.dataset_path, "label.txt"))

    # Load and preprocess dataset.
    train_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.train_file,
        max_seq_len=training_args.max_seq_length,
        lazy=False,
    )
    dev_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.dev_file,
        lazy=False,
    )

    test_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.test_file,
        lazy=False,
    )

    # Define the criterion.
    criterion = UTCLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics_single_label(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.softmax(preds, axis=-1)
        labels = paddle.argmax(labels, axis=-1)
        metric = Accuracy()
        correct = metric.compute(preds, labels)
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    def compute_metrics_paddle(eval_preds):
        metric = MetricReport()
        preds = paddle.to_tensor(eval_preds.predictions)
        metric.reset()
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
        micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
        metric.reset()
        return {
            "eval_micro_f1": micro_f1_score,
            "eval_macro_f1": macro_f1_score,
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
        }

    def compute_metrics_sklearn(eval_preds):
        separate_eval = True
        metric = MetricReport()
        metric.reset()

        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.sigmoid(preds)

        logger.debug(preds)

        # logger.debug(f"Prediction: {np.where(preds[labels != -100] > data_args.threshold)}")
        # logger.debug(f"Ground True: {np.where(labels[labels != -100])}")

        # logger.debug(
        #    f"not the same: {np.where((preds[labels != -100] > data_args.threshold) != labels[labels != -100])}"
        # )

        # breakpoint()
        # preds = preds[labels != -100]
        # labels = labels[labels != -100]
        preds = preds > data_args.threshold

        logger.info(f"Number of All True 1 labels: {paddle.sum(labels==1).item()}")
        logger.info(f"Number of All Predict 1 label: {paddle.sum(preds).item()}")
        if labels.shape[1] != 55:
            logger.warning(
                "Cannot apply separate evaluate. Due to the number of labels is not equal to 55. (Add or Remove labels ever?)"
            )
            separate_eval = False

        if separate_eval:
            label_name = ["體傷部位", "體傷程度", "肇事責任", "年齡"]
            # 部位: 0~8
            preds_injure_part = preds[:, :9]
            labels_injure_part = labels[:, :9]

            # 傷勢: 9 ~ 35
            preds_injure_level = preds[:, 9:36]
            labels_injure_level = labels[:, 9:36]

            # 肇事: 36 ~ 46
            preds_responsibility = preds[:, 36:47]
            labels_responsibility = labels[:, 36:47]

            # 年齡: 47 ~ 54
            preds_age = preds[:, 47:]
            labels_age = labels[:, 47:]

            preds_list = (preds_injure_part, preds_injure_level, preds_responsibility, preds_age)
            labels_list = (labels_injure_part, labels_injure_level, labels_responsibility, labels_age)

            for p, l, name in zip(preds_list, labels_list, label_name):
                metric.update(p, l)
                micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
                logger.debug(f"====={name}=====")
                logger.debug(
                    f"micro_f1_score: {micro_f1_score}. macro_f1_score: {macro_f1_score}. accuracy: {accuracy}. precision: {precision}. recall: {recall}."
                )
                metric.reset()

        metric.update(preds, labels)
        micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
        metric.reset()
        return {
            "eval_micro_f1": micro_f1_score,
            "eval_macro_f1": macro_f1_score,
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
        }

    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=compute_metrics_sklearn,
    )
    # compute_metrics=compute_metrics_single_label if data_args.single_label else compute_metrics,
    # data_collator=myDataCollator(tokenizer, padding=True, return_tensors="pd"),

    # Training.
    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        trainer.log_metrics("test", test_metrics)

    # Export.
    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
            InputSpec(shape=[None, None], dtype="int64", name="omask_positions"),
            InputSpec(shape=[None], dtype="int64", name="cls_positions"),
        ]
        export_model(trainer.pretrained_model, input_spec, model_args.export_model_dir, model_args.export_type)


if __name__ == "__main__":
    import os
    
    main()
