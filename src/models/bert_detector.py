from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class BertSteganalysisDetector(nn.Module):
    """
    BERT-based binary steganalysis detector.

    Labels:
        0 = cover/original text
        1 = stego text
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
    ):
        super().__init__()

        self.model_name = model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass.

        During training:
            labels are provided and the model calculates loss.

        During prediction:
            labels can be None.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Hugging Face may type outputs.loss as Optional[Tensor].
        # During training, it should be a Tensor.
        if outputs.loss is None:
            loss = torch.tensor(
                0.0,
                device=outputs.logits.device,
            )
        else:
            loss = outputs.loss

        return {
            "loss": loss,
            "logits": outputs.logits,
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted labels and class probabilities.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1,
        )

        return predictions, probabilities