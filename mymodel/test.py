from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoModel, AutoTokenizer, ErnieConfig, ErniePretrainedModel, ErnieModel
from paddle import nn, squeeze, Tensor
from paddle.static import InputSpec

MODEL_NAME = "ernie-3.0-base-zh"


# model =
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class myModel(ErniePretrainedModel):
    """
    Ernie Model with two linear layer on top of the hidden-states
    output to compute `start_prob` and `end_prob`,
    designed for Universal Information Extraction.
    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct UIE
    """

    def __init__(self, config: ErnieConfig):
        super(myModel, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
        ]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import UIE, ErnieTokenizer
                tokenizer = ErnieTokenizer.from_pretrained('uie-base')
                model = UIE.from_pretrained('uie-base')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                start_prob, end_prob = model(**inputs)
        """

        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob


class myModel(ErniePretrainedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
