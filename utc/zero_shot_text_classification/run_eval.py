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
from dataclasses import dataclass, field
from paddlenlp.utils.log import logger
from metric import MetricReport

import paddle
from paddle.metric import Accuracy
from sklearn.metrics import f1_score
from utils import UTCLoss, read_local_dataset

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import UTC, AutoTokenizer


@dataclass
class DataArguments:
    test_path: str = field(default="./data/test.txt", metadata={"help": "Test dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})
    single_label: str = field(default=False, metadata={"help": "Predict exactly one label per sample."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="utc-base", metadata={"help": "Build-in pretrained model."})
    model_path: str = field(default=None, metadata={"help": "Build-in pretrained model."})


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

    # Load and preprocess dataset.
    if data_args.test_path is not None:
        test_ds = load_dataset(read_local_dataset, data_path=data_args.test_path, lazy=False)

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )
    if model_args.model_path is not None:
        model_state = paddle.load(os.path.join(model_args.model_path, "model_state.pdparams"))
        prompt_model.set_state_dict(model_state)

    # Define the metric function.
    def compute_metrics_single_label(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.softmax(preds, axis=-1)
        labels = paddle.argmax(labels, axis=-1)
        print(preds, labels)
        metric = Accuracy()
        correct = metric.compute(preds, labels)
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    def compute_metrics(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)

        preds = paddle.nn.functional.sigmoid(preds)
        preds = preds[labels != -100].numpy()
        labels = labels[labels != -100].numpy()
        preds = preds > data_args.threshold
        micro_f1 = f1_score(y_pred=preds, y_true=labels, average="micro")
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average="macro")

        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

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
        criterion=UTCLoss(),
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        compute_metrics=compute_metrics_single_label if data_args.single_label else compute_metrics_sklearn,
    )

    if data_args.test_path is not None:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        with open(os.path.join(training_args.output_dir, "test_metric.json"), "w", encoding="utf-8") as fp:
            json.dump(test_ret.metrics, fp)

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
