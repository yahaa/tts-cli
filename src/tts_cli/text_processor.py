"""Text processing utilities for tts-cli.

This module handles text splitting and normalization for ChatTTS.
Key consideration: ChatTTS produces lower quality audio for long texts,
so intelligent splitting is critical.
"""

import re
from typing import List

# Default maximum length for each chunk (characters)
DEFAULT_MAX_LENGTH = 800

# Minimum length for the last chunk to avoid quality issues
DEFAULT_MIN_TAIL_LENGTH = 300

# Sentence ending punctuation patterns
SENTENCE_END_PATTERN_EN = r'[.!?]'
SENTENCE_END_PATTERN_ZH = r'[。！？]'
SENTENCE_END_PATTERN = r'[.!?。！？]'


def normalize_text_for_tts(text: str) -> str:
    """
    Normalize text for ChatTTS compatibility.

    ChatTTS has a very limited character set. This function aggressively
    converts or removes unsupported characters to prevent generation failures.

    Confirmed supported:
    - English letters (a-z, A-Z)
    - Chinese characters (U+4E00-U+9FFF)
    - Chinese punctuation: 。，！？
    - Space

    Args:
        text: Input text

    Returns:
        Normalized text safe for ChatTTS
    """
    # Step 1: Convert numbers to words
    num_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    # Replace common year patterns
    text = text.replace('2026', 'twenty twenty six')
    text = text.replace('2025', 'twenty twenty five')
    text = text.replace('2024', 'twenty twenty four')
    text = text.replace('2023', 'twenty twenty three')

    # Replace multi-digit numbers
    def replace_number(match):
        num = int(match.group(0))
        if num <= 20:
            words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                    'nineteen', 'twenty']
            return words[num]
        return ' '.join(num_to_word[d] for d in str(num))

    text = re.sub(r'\b\d+\b', replace_number, text)

    # Step 2: Convert punctuation - remove ? entirely as ChatTTS doesn't support it
    punctuation_map = {
        # Question marks - REMOVE (ChatTTS reports "found invalid characters: {'?'}")
        '?': '，',   # Replace with comma for a pause
        '？': '，',  # Full-width too
        # Half-width to full-width
        '.': '。',
        ',': '，',
        '!': '！',
        ';': '，',
        ':': '，',
        # Ellipsis
        '…': '。',
        '⋯': '。',
    }

    for old, new in punctuation_map.items():
        text = text.replace(old, new)

    # Step 3: Only keep confirmed supported characters
    result = []
    for char in text:
        if char.isascii() and char.isalpha():
            # English letters a-z, A-Z
            result.append(char)
        elif '\u4e00' <= char <= '\u9fff':
            # Chinese characters
            result.append(char)
        elif char in '。，！':
            # Chinese punctuation (NO question mark - not supported)
            result.append(char)
        elif char == ' ' or char == '\n':
            # Whitespace
            result.append(' ')
        else:
            # Everything else becomes space
            result.append(' ')

    text = ''.join(result)

    # Step 4: Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def validate_text_for_chattts(text: str) -> tuple:
    """
    Validate that text only contains ChatTTS-supported characters.

    Returns:
        Tuple of (is_valid, invalid_chars)
    """
    invalid_chars = set()
    for char in text:
        if char.isascii() and char.isalpha():
            continue  # a-z, A-Z
        elif '\u4e00' <= char <= '\u9fff':
            continue  # Chinese
        elif char in '。，！？ ':
            continue  # Allowed punctuation and space
        else:
            invalid_chars.add(char)

    return (len(invalid_chars) == 0, invalid_chars)


def split_text_intelligently(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_tail_length: int = DEFAULT_MIN_TAIL_LENGTH
) -> List[str]:
    """
    Split long text into smaller chunks by paragraphs and sentences.

    Strategy:
    1. First split by paragraphs (one or more newlines)
    2. If a paragraph exceeds max_length, split at sentence endings
    3. If the last chunk is too short, merge with previous chunk

    IMPORTANT: ChatTTS produces lower quality audio for long texts.
    Default max_length=800 is recommended for best quality.

    Args:
        text: Input text to split
        max_length: Maximum length of each chunk (default: 800)
        min_tail_length: Minimum length for the last chunk (default: 300)

    Returns:
        List of text chunks, each <= max_length characters
    """
    # If text is short enough, return as is
    if len(text) <= max_length:
        return [text]

    # Step 1: Split by paragraphs (one or more newlines)
    paragraphs = re.split(r'\n+', text)

    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph is short enough, add directly
        if len(para) <= max_length:
            chunks.append(para)
            continue

        # Step 2: Paragraph too long, split at sentence endings
        remaining = para
        while len(remaining) > max_length:
            # Find the last sentence ending within max_length
            search_text = remaining[:max_length]
            last_end = _find_last_sentence_end(search_text)

            if last_end > 0:
                # Split at the last sentence ending
                chunk = remaining[:last_end + 1].strip()
                remaining = remaining[last_end + 1:].strip()
                chunks.append(chunk)
            else:
                # No sentence ending found, force split at max_length
                chunks.append(remaining[:max_length].strip())
                remaining = remaining[max_length:].strip()

        # Add remaining text
        if remaining:
            chunks.append(remaining)

    # Step 3: Merge short tail chunk with previous chunk
    if len(chunks) > 1 and len(chunks[-1]) < min_tail_length:
        last_chunk = chunks.pop()
        chunks[-1] = chunks[-1] + " " + last_chunk

    return chunks


def _find_last_sentence_end(text: str) -> int:
    """
    Find the position of the last sentence ending punctuation.

    Supports both English (. ! ?) and Chinese (。！？) punctuation.

    Args:
        text: Text to search

    Returns:
        Position of last sentence ending, or -1 if not found
    """
    # Find all sentence ending positions
    positions = []
    for match in re.finditer(SENTENCE_END_PATTERN, text):
        positions.append(match.end() - 1)

    return positions[-1] if positions else -1


def split_paragraph_to_sentences(text: str) -> List[str]:
    """
    Split a paragraph into sentences.

    Supports both English and Chinese sentence endings.

    Args:
        text: Input paragraph text

    Returns:
        List of sentences
    """
    # Split at sentence endings, handling both:
    # 1. Punctuation followed by whitespace (English style)
    # 2. Chinese punctuation (may not have following whitespace)
    # Using lookbehind to keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])', text)
    return [s.strip() for s in sentences if s.strip()]


# Sentence separator - just use space, punctuation already provides natural pauses
# [uv_break] contains invalid characters ([, ], _) that ChatTTS rejects
CHATTTS_PAUSE_MARKER = ' '


def merge_sentences_to_chunks(
    sentences: List[str],
    target_length: int = 600,
    max_length: int = 800,
    min_sentence_length: int = 10
) -> List[str]:
    """
    Merge sentences into larger chunks for more efficient TTS processing.

    Short sentences are merged with adjacent sentences using ChatTTS pause markers.
    Each chunk aims to be close to target_length but not exceed max_length.

    Args:
        sentences: List of sentences to merge
        target_length: Target chunk length (default: 600)
        max_length: Maximum chunk length (default: 800)
        min_sentence_length: Minimum length for a standalone sentence (default: 10)

    Returns:
        List of merged chunks with pause markers
    """
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)
        # Account for pause marker length when merging
        marker_len = len(CHATTTS_PAUSE_MARKER) if current_chunk else 0
        new_length = current_length + marker_len + sentence_len

        if current_chunk and new_length > max_length:
            # Current chunk is full, save it and start new one
            chunks.append(CHATTTS_PAUSE_MARKER.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        elif current_chunk and current_length >= target_length and sentence_len >= min_sentence_length:
            # Current chunk reached target and next sentence is long enough to stand alone
            chunks.append(CHATTTS_PAUSE_MARKER.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length = new_length

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(CHATTTS_PAUSE_MARKER.join(current_chunk))

    return chunks


def split_and_merge_text(
    text: str,
    target_length: int = 600,
    max_length: int = 800
) -> List[str]:
    """
    Split text into sentences and merge them into optimal chunks.

    This is the recommended function for preparing text for ChatTTS.
    It ensures chunks are close to target_length for optimal quality
    and includes pause markers between sentences.

    Args:
        text: Input text
        target_length: Target chunk length (default: 600)
        max_length: Maximum chunk length (default: 800)

    Returns:
        List of text chunks ready for TTS
    """
    # First split by paragraphs
    paragraphs = re.split(r'\n+', text)

    all_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split paragraph into sentences
        sentences = split_paragraph_to_sentences(para)
        if not sentences:
            sentences = [para]

        # Merge sentences into chunks
        chunks = merge_sentences_to_chunks(
            sentences,
            target_length=target_length,
            max_length=max_length
        )
        all_chunks.extend(chunks)

    return all_chunks


def estimate_text_length(text: str) -> int:
    """
    Estimate the "effective" length of text for TTS.

    Chinese characters are typically shorter in audio duration than English words,
    but for simplicity we use character count.

    Args:
        text: Input text

    Returns:
        Estimated length
    """
    return len(text)


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Chinese or English.

    Args:
        text: Input text

    Returns:
        'zh' for Chinese, 'en' for English
    """
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return 'en'

    # If more than 30% Chinese characters, consider it Chinese
    if chinese_chars / total_chars > 0.3:
        return 'zh'
    return 'en'
