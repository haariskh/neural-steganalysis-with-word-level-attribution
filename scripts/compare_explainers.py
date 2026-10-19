
from __future__ import annotations

import json
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CHECKPOINT_DIR, EXPLANATION_DIR, PROCESSED_DATA_DIR


def normalize_word(value):
    """Normalize a token for word-level comparison."""
    value = str(value).strip().lower()
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", value)
    if value.startswith("##"):
        value = value[2:]
    return value


def unique_ranked_words(items, word_key, score_key):
    """Convert scored tokens into a unique, ranked word list."""
    ranked = sorted(
        items,
        key=lambda item: abs(float(item.get(score_key, 0))),
        reverse=True,
    )
    result = []
    seen = set()

    for item in ranked:
        word = normalize_word(item.get(word_key, ""))
        if len(word) < 2 or not re.search(r"[a-z0-9]", word):
            continue
        if word not in seen:
            result.append(word)
            seen.add(word)
    return result


def read_explanations(base_dir, index, predicted_label):
    """Read existing IG, LIME, and SHAP results."""
    filename = f"explanation_{index:05d}.json"

    ig_path = base_dir / "integrated_gradients" / filename
    lime_path = base_dir / "lime" / filename
    shap_path = base_dir / "shap" / filename

    for path in (ig_path, lime_path, shap_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing explanation file: {path}\n"
                "Run python -m scripts.explain_all first."
            )

    ig = json.loads(ig_path.read_text(encoding="utf-8"))
    lime = json.loads(lime_path.read_text(encoding="utf-8"))
    shap_data = json.loads(shap_path.read_text(encoding="utf-8"))

    ig_words = unique_ranked_words(
        ig["words"], "word", "attribution"
    )

    class_explanations = lime["class_explanations"]
    class_items = class_explanations.get(str(predicted_label), [])
    lime_words = unique_ranked_words(
        class_items, "word", "weight"
    )

    shap_tokens = shap_data["tokens"]
    shap_scores = shap_data.get("target_attributions")

    # Support the earlier SHAP JSON format if present.
    if shap_scores is None:
        values = np.asarray(shap_data["values"])
        if values.ndim == 3:
            values = values[0]
        shap_scores = values[:, predicted_label].tolist()

    if len(shap_tokens) != len(shap_scores):
        raise ValueError(
            "SHAP token/attribution counts do not match: "
            f"{len(shap_tokens)} vs {len(shap_scores)}"
        )

    shap_items = [
        {"word": token, "score": score}
        for token, score in zip(shap_tokens, shap_scores)
    ]
    shap_words = unique_ranked_words(
        shap_items, "word", "score"
    )

    return {
        "IG": ig_words,
        "LIME": lime_words,
        "SHAP": shap_words,
    }


def replace_words(text, words, mask_token):
    """Replace complete word matches with the model's mask token."""
    result = text
    for word in words:
        if not word:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(word)}(?!\w)",
            flags=re.IGNORECASE,
        )
        result = pattern.sub(f" {mask_token} ", result)
    return result


def jaccard(left, right):
    """Jaccard similarity between two word sets."""
    a, b = set(left), set(right)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare IG, LIME, and SHAP explanations."
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    checkpoint = Path(CHECKPOINT_DIR) / "bert_detector"
    dataset_path = (
        Path(PROCESSED_DATA_DIR) / "test_binary_detector.csv"
    )
    explanation_root = Path(EXPLANATION_DIR)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if not 0 <= args.index < len(df):
        raise IndexError(
            f"Invalid index {args.index}; valid range is 0 to {len(df)-1}"
        )

    row = df.iloc[args.index]
    text = str(row["review"])
    actual_label = int(row["is_stego"])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]

        outputs = []
        for start in range(0, len(texts), 4):
            batch = [str(x) for x in texts[start:start + 4]]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=-1)
            outputs.extend(probs.cpu().numpy())

        return np.asarray(outputs)

    start = time.perf_counter()
    probabilities = predict_proba([text])[0]
    prediction_time = time.perf_counter() - start
    predicted_label = int(np.argmax(probabilities))

    explanations = read_explanations(
        explanation_root, args.index, predicted_label
    )
    top_words = {
        method: words[:args.top_k]
        for method, words in explanations.items()
    }

    print("\nPrediction")
    print(f"Actual class: {'stego' if actual_label else 'cover'}")
    print(f"Predicted class: {'stego' if predicted_label else 'cover'}")
    print(f"Cover probability: {probabilities[0]:.4f}")
    print(f"Stego probability: {probabilities[1]:.4f}")

    print(f"\nTop {args.top_k} words by method")
    for method, words in top_words.items():
        print(f"{method}: {', '.join(words) if words else '(no words found)'}")

    # Pairwise overlap between the top-ranked words.
    overlap_results = []
    for left, right in combinations(top_words.keys(), 2):
        overlap = jaccard(top_words[left], top_words[right])
        overlap_results.append({
            "method_a": left,
            "method_b": right,
            "top_k": args.top_k,
            "shared_word_count": len(
                set(top_words[left]) & set(top_words[right])
            ),
            "jaccard_overlap": overlap,
        })

    # Deletion test: replace important words and compare probabilities.
    # This is a perturbation-based proxy, not a definitive proof of
    # explanation faithfulness. The mask token may be out-of-distribution.
    mask_token = tokenizer.mask_token or "[MASK]"
    deletion_results = []

    baseline_probability = float(probabilities[predicted_label])

    for method, ranked in explanations.items():
        high_words = ranked[:args.top_k]
        low_words = ranked[-args.top_k:] if ranked else []

        for group_name, words in (
            ("top_attributed", high_words),
            ("bottom_attributed", low_words),
        ):
            modified_text = replace_words(text, words, mask_token)
            modified_probability = float(
                predict_proba([modified_text])[0][predicted_label]
            )

            deletion_results.append({
                "method": method,
                "group": group_name,
                "words_replaced": words,
                "baseline_predicted_class_probability": baseline_probability,
                "modified_predicted_class_probability": modified_probability,
                "probability_drop": baseline_probability - modified_probability,
            })

    output_dir = explanation_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_index": args.index,
        "top_k": args.top_k,
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "probabilities": {
            "cover": float(probabilities[0]),
            "stego": float(probabilities[1]),
        },
        "prediction_time_seconds": prediction_time,
        "top_words": top_words,
        "pairwise_top_word_overlap": overlap_results,
        "deletion_test": deletion_results,
        "limitations": [
            "Results are for one review unless multiple indices are evaluated.",
            "Deletion uses mask-token replacement, which may create unnatural text.",
            "Top-word overlap depends on word normalization and token alignment.",
            "This script does not measure ground-truth synonym-change coverage.",
            "The three explainers do not have separate classification accuracies.",
        ],
    }

    json_path = output_dir / f"comparison_{args.index:05d}.json"
    csv_path = output_dir / f"comparison_{args.index:05d}.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    flat_rows = []
    for item in overlap_results:
        flat_rows.append({
            "section": "top_word_overlap",
            **item,
        })
    for item in deletion_results:
        flat_rows.append({
            "section": "deletion_test",
            **item,
        })
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print("\nPairwise top-word overlap (Jaccard)")
    for item in overlap_results:
        print(
            f"{item['method_a']} vs {item['method_b']}: "
            f"{item['jaccard_overlap']:.3f} "
            f"({item['shared_word_count']} shared words)"
        )

    print("\nDeletion test")
    for item in deletion_results:
        print(
            f"{item['method']} / {item['group']}: "
            f"probability drop = {item['probability_drop']:+.4f}"
        )

    print(f"\nJSON report saved: {json_path}")
    print(f"CSV report saved: {csv_path}")
    print(
        "\nImportant: a larger probability drop is evidence from this "
        "perturbation test, not proof that an explainer is universally better."
    )


if __name__ == "__main__":
    main()
