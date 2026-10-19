from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for a Hugging Face
    sequence-classification model.

    Class labels:
        0 = cover
        1 = synonym-stego
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        num_steps: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_steps = num_steps

        self.model.eval()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get BERT's input embedding vectors.

        The Hugging Face API is dynamically typed, so the model
        is intentionally treated as Any.
        """

        embedding_layer = self.model.get_input_embeddings()

        embeddings = embedding_layer(input_ids)

        return embeddings

    def _forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the model using input embeddings.
        """

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

        return outputs.logits

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Integrated Gradients for one text.

        Args:
            text:
                Input review.

            target_class:
                Class to explain.
                If None, explain the predicted class.

        Returns:
            Prediction information and token/word attributions.
        """

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        input_embeddings = self._get_embeddings(
            input_ids
        ).detach()

        # Zero embedding baseline.
        baseline_embeddings = torch.zeros_like(
            input_embeddings
        )

        # Obtain the model's original prediction.
        with torch.no_grad():
            original_logits = self._forward_from_embeddings(
                input_embeddings,
                attention_mask,
            )

            probabilities = torch.softmax(
                original_logits,
                dim=-1,
            )[0]

            predicted_class = int(
                torch.argmax(probabilities).item()
            )

        if target_class is None:
            target_class = predicted_class

        if target_class not in [0, 1]:
            raise ValueError(
                "target_class must be 0, 1, or None."
            )

        # Accumulate gradients along the interpolation path.
        total_gradients = torch.zeros_like(
            input_embeddings
        )

        for step in range(1, self.num_steps + 1):
            alpha = step / self.num_steps

            interpolated_embeddings = (
                baseline_embeddings
                + alpha
                * (
                    input_embeddings
                    - baseline_embeddings
                )
            ).detach()

            interpolated_embeddings.requires_grad_(True)

            logits = self._forward_from_embeddings(
                interpolated_embeddings,
                attention_mask,
            )

            target_logit = logits[0, target_class]

            self.model.zero_grad(
                set_to_none=True
            )

            gradients = torch.autograd.grad(
                outputs=target_logit,
                inputs=interpolated_embeddings,
                retain_graph=False,
                create_graph=False,
            )[0]

            total_gradients += gradients.detach()

        average_gradients = (
            total_gradients / self.num_steps
        )

        # Integrated Gradients:
        # (input - baseline) * average gradient
        integrated_gradients = (
            input_embeddings - baseline_embeddings
        ) * average_gradients

        # Sum over the embedding dimension to get one
        # attribution score per token.
        token_attributions = integrated_gradients.sum(
            dim=-1
        )[0]

        token_ids = (
            input_ids[0]
            .detach()
            .cpu()
            .tolist()
        )

        attention_values = (
            attention_mask[0]
            .detach()
            .cpu()
            .tolist()
        )

        attribution_values = (
            token_attributions
            .detach()
            .cpu()
            .tolist()
        )

        # Hugging Face tokenizer methods are dynamically typed.
        tokens = self.tokenizer.convert_ids_to_tokens(
            token_ids
        )

        special_tokens = set(
            self.tokenizer.all_special_tokens
        )

        token_results: List[Dict[str, Any]] = []

        for token, attention_value, attribution in zip(
            tokens,
            attention_values,
            attribution_values,
        ):
            # Remove padding tokens.
            if attention_value == 0:
                continue

            # Remove [CLS], [SEP], [PAD], etc.
            if token in special_tokens:
                continue

            token_results.append(
                {
                    "token": str(token),
                    "attribution": float(attribution),
                }
            )

        word_results = self._aggregate_subwords(
            token_results
        )

        return {
            "text": text,
            "target_class": int(target_class),
            "target_class_name": self._class_name(
                target_class
            ),
            "predicted_class": predicted_class,
            "predicted_class_name": self._class_name(
                predicted_class
            ),
            "cover_probability": float(
                probabilities[0].item()
            ),
            "stego_probability": float(
                probabilities[1].item()
            ),
            "num_steps": self.num_steps,
            "tokens": token_results,
            "words": word_results,
        }

    def _aggregate_subwords(
        self,
        token_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine BERT wordpiece tokens into complete words.

        Example:
            un + ##believable
        becomes:
            unbelievable
        """

        words: List[Dict[str, Any]] = []

        for item in token_results:
            token = str(item["token"])
            attribution = float(item["attribution"])

            if token.startswith("##") and words:
                words[-1]["word"] += token[2:]
                words[-1]["attribution"] += attribution
            else:
                words.append(
                    {
                        "word": token,
                        "attribution": attribution,
                    }
                )

        return words

    @staticmethod
    def _class_name(class_id: int) -> str:
        if class_id == 0:
            return "cover"

        if class_id == 1:
            return "synonym-stego"

        return f"class-{class_id}"

    @staticmethod
    def save_json(
        explanation: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """
        Save the explanation as a JSON file.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                explanation,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def save_html(
        explanation: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """
        Save a color-coded HTML visualization.

        Positive attribution:
            pushes toward the explained class.

        Negative attribution:
            pushes away from the explained class.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        words = explanation["words"]

        max_abs_attribution = max(
            [
                abs(float(item["attribution"]))
                for item in words
            ],
            default=1.0,
        )

        html_words: List[str] = []

        for item in words:
            word = str(item["word"])
            attribution = float(
                item["attribution"]
            )

            normalized = min(
                abs(attribution)
                / max_abs_attribution,
                1.0,
            )

            if attribution >= 0:
                background = (
                    "rgba(255, 80, 80, "
                    f"{normalized:.3f})"
                )
            else:
                background = (
                    "rgba(80, 130, 255, "
                    f"{normalized:.3f})"
                )

            html_words.append(
                (
                    f'<span title="Attribution: '
                    f'{attribution:.6f}" '
                    f'style="background-color: '
                    f'{background}; padding: 3px; '
                    f'margin: 2px; display: '
                    f'inline-block;">'
                    f"{word}</span>"
                )
            )

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Integrated Gradients Explanation</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 2.2;
            margin: 35px;
        }}

        .metadata {{
            background: #f2f2f2;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}

        .text {{
            font-size: 18px;
        }}

        .legend {{
            margin-top: 25px;
            font-size: 14px;
        }}
    </style>
</head>

<body>

<h1>Integrated Gradients Explanation</h1>

<div class="metadata">
    <p>
        <strong>Predicted class:</strong>
        {explanation["predicted_class_name"]}
    </p>

    <p>
        <strong>Cover probability:</strong>
        {explanation["cover_probability"]:.4f}
    </p>

    <p>
        <strong>Stego probability:</strong>
        {explanation["stego_probability"]:.4f}
    </p>

    <p>
        <strong>Explained target:</strong>
        {explanation["target_class_name"]}
    </p>
</div>

<h2>Word-level attribution</h2>

<div class="text">
    {" ".join(html_words)}
</div>

<div class="legend">
    <p>
        <strong>Red:</strong>
        pushes the prediction toward the explained class.
    </p>

    <p>
        <strong>Blue:</strong>
        pushes the prediction away from the explained class.
    </p>

    <p>
        Hover over a word to see its attribution score.
    </p>
</div>

</body>
</html>
"""

        output_path.write_text(
            html_content,
            encoding="utf-8",
        )