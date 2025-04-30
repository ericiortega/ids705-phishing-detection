import random
import re

#  Cyrillic/Greek characters and symbols that will be used for adversarial character noise
homoglyph_map = {
    "a": "а",  # Cyrillic a
    "e": "е",  # Cyrillic e
    "i": "і",  # Cyrillic i
    "o": "о",  # Cyrillic o
    "c": "с",  # Cyrillic c
    "p": "р",  # Cyrillic p
    "s": "$",  # Spam symbol
    "b": "Ь",  # Cyrillic soft b
    "d": "ԁ",  # Cyrillic d
    "t": "τ",  # Greek tau
}


def add_homoglyph_noise(text, noise_level=0.05, seed=None):
    """
    Replace visually identifiable characters with homoglyphs at a given probability.

    Args:
        text (str): The original input text.
        noise_level (float): Probability of replacing a character.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        str: The perturbed text.
    """
    if seed is not None:
        random.seed(seed)

    noisy = ""
    for char in text:
        if char.lower() in homoglyph_map and random.random() < noise_level:
            noisy += homoglyph_map[char.lower()]
        else:
            noisy += char
    return noisy


def add_grammar_noise(text, noise_level=0.05, seed=None):
    """
    Randomly changes casing and removes punctuation from the text.

    Args:
        text (str): The input string.
        noise_level (float): Probability of altering each alphabetical character.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        str: The text with grammar noise.
    """
    if seed is not None:
        random.seed(seed)

    text = re.sub(r"[.,!?;:]", "", text)  # remove punctuation
    noisy = ""
    for char in text:
        if char.isalpha() and random.random() < noise_level:
            noisy += char.upper() if random.random() < 0.5 else char.lower()
        else:
            noisy += char
    return noisy


def add_combined_noise(text, noise_level=0.05, seed=None):
    """
    Applies both homoglyph and grammar noise to a given string.

    Args:
        text (str): The input string.
        noise_level (float): Noise application probability.
        seed (int, optional): Random seed.

    Returns:
        str: The corrupted text.
    """
    text = add_homoglyph_noise(text, noise_level, seed)
    text = add_grammar_noise(text, noise_level, seed)
    return text
