from src.steganography.synonym_stega import SynonymSteganography


def main():
    encoder = SynonymSteganography(
        max_substitutions=5,
        substitution_probability=0.5,
        seed=42,
    )

    text = (
        "This movie was absolutely fantastic and the performances "
        "were excellent. The story was engaging and emotionally powerful."
    )

    stego_text, metadata = encoder.encode(text)

    print("Original:")
    print(text)

    print("\nStego:")
    print(stego_text)

    print("\nMetadata:")
    print(metadata)


if __name__ == "__main__":
    main()