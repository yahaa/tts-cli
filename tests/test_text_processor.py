"""Tests for text_processor module."""

import pytest
from tts_cli.text_processor import (
    split_text_intelligently,
    split_paragraph_to_sentences,
    normalize_text_for_tts,
    detect_language,
    _find_last_sentence_end,
)


class TestSplitTextIntelligently:
    """Tests for split_text_intelligently function."""

    def test_short_text_no_split(self):
        """Short text should not be split."""
        text = "This is a short text."
        result = split_text_intelligently(text, max_length=800)
        assert result == [text]

    def test_split_by_paragraph(self):
        """Text should be split by paragraphs when exceeding max_length."""
        # Create text where each paragraph exceeds max_length individually
        text = "First paragraph. " + "A" * 100 + "\n\nSecond paragraph. " + "B" * 100
        result = split_text_intelligently(text, max_length=80)
        assert len(result) >= 2
        # First chunk should contain "First paragraph"
        assert "First paragraph" in result[0]

    def test_split_long_paragraph_at_sentence(self):
        """Long paragraph should be split at sentence boundaries."""
        # Create a paragraph with multiple sentences
        sentences = ["This is sentence one."] * 20
        text = " ".join(sentences)
        result = split_text_intelligently(text, max_length=100)

        # Should have multiple chunks
        assert len(result) > 1
        # Each chunk (except possibly the last merged one) should respect sentence boundaries
        for chunk in result[:-1]:
            assert chunk.endswith('.') or chunk.endswith('!') or chunk.endswith('?')

    def test_split_chinese_text(self):
        """Chinese text should be split at Chinese punctuation."""
        text = "这是第一句话。这是第二句话。这是第三句话。" * 50
        result = split_text_intelligently(text, max_length=100)

        # Each chunk should respect the max_length
        for chunk in result[:-1]:  # Last chunk may be merged
            assert len(chunk) <= 100 or "。" not in chunk[:100]

    def test_merge_short_tail(self):
        """Short tail chunk should be merged with previous."""
        text = "A" * 700 + ". " + "B" * 50
        result = split_text_intelligently(text, max_length=800, min_tail_length=300)

        # Should be a single chunk since tail is too short
        assert len(result) == 1

    def test_exact_max_length(self):
        """Text exactly at max_length should not be split."""
        text = "A" * 800
        result = split_text_intelligently(text, max_length=800)
        assert len(result) == 1

    def test_empty_text(self):
        """Empty text should return empty list."""
        result = split_text_intelligently("", max_length=800)
        assert result == [""]

    def test_no_punctuation_force_split(self):
        """Text without punctuation should be force-split at max_length."""
        text = "A" * 2000
        result = split_text_intelligently(text, max_length=800)
        assert len(result) >= 2


class TestSplitParagraphToSentences:
    """Tests for split_paragraph_to_sentences function."""

    def test_english_sentences(self):
        """English sentences should be split correctly."""
        text = "First sentence. Second sentence! Third sentence?"
        result = split_paragraph_to_sentences(text)
        assert len(result) == 3
        assert "First sentence." in result[0]
        assert "Second sentence!" in result[1]
        assert "Third sentence?" in result[2]

    def test_chinese_sentences(self):
        """Chinese sentences should be split correctly."""
        text = "第一句。第二句！第三句？"
        result = split_paragraph_to_sentences(text)
        assert len(result) == 3

    def test_mixed_language(self):
        """Mixed language text should be split correctly."""
        text = "Hello world. 你好世界。How are you?"
        result = split_paragraph_to_sentences(text)
        assert len(result) == 3

    def test_no_punctuation(self):
        """Text without punctuation should return as single item."""
        text = "No punctuation here"
        result = split_paragraph_to_sentences(text)
        assert result == ["No punctuation here"]


class TestFindLastSentenceEnd:
    """Tests for _find_last_sentence_end function."""

    def test_find_english_period(self):
        """Should find English period."""
        text = "Hello world. How are you"
        pos = _find_last_sentence_end(text)
        assert text[pos] == "."

    def test_find_chinese_period(self):
        """Should find Chinese period."""
        text = "你好世界。今天好吗"
        pos = _find_last_sentence_end(text)
        assert text[pos] == "。"

    def test_find_exclamation(self):
        """Should find exclamation mark."""
        text = "Hello! How are you"
        pos = _find_last_sentence_end(text)
        assert text[pos] == "!"

    def test_no_sentence_end(self):
        """Should return -1 when no sentence end found."""
        text = "No sentence end here"
        pos = _find_last_sentence_end(text)
        assert pos == -1


class TestNormalizeTextForTts:
    """Tests for normalize_text_for_tts function."""

    def test_number_to_word(self):
        """Single digits should be converted to words."""
        text = "I have 3 apples"
        result = normalize_text_for_tts(text)
        assert "three" in result

    def test_year_conversion(self):
        """Years should be converted to words."""
        text = "In 2024"
        result = normalize_text_for_tts(text)
        assert "twenty twenty four" in result

    def test_remove_special_quotes(self):
        """Special quotes should be removed or simplified."""
        text = '"Hello"'
        result = normalize_text_for_tts(text)
        assert '"' not in result
        assert '"' not in result

    def test_em_dash_to_space(self):
        """Em-dash should be converted to space."""
        text = "Hello—world"
        result = normalize_text_for_tts(text)
        assert "—" not in result
        assert "Hello" in result
        assert "world" in result

    def test_hyphen_to_space(self):
        """Hyphen should be converted to space."""
        text = "command-line"
        result = normalize_text_for_tts(text)
        assert "-" not in result
        assert "command" in result
        assert "line" in result

    def test_multiple_spaces(self):
        """Multiple spaces should be collapsed."""
        text = "Hello    world"
        result = normalize_text_for_tts(text)
        assert "    " not in result

    def test_only_allowed_chars_remain(self):
        """Only allowed characters should remain."""
        text = "Hello @world #test 123!"
        result = normalize_text_for_tts(text)
        assert "@" not in result
        assert "#" not in result
        # Numbers should be converted to words
        assert "one two three" in result

    def test_chinese_text_preserved(self):
        """Chinese characters should be preserved."""
        text = "你好世界"
        result = normalize_text_for_tts(text)
        assert result == "你好世界"


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_english_text(self):
        """English text should be detected as 'en'."""
        text = "Hello, how are you today?"
        assert detect_language(text) == "en"

    def test_chinese_text(self):
        """Chinese text should be detected as 'zh'."""
        text = "你好，今天怎么样？"
        assert detect_language(text) == "zh"

    def test_mixed_mostly_english(self):
        """Mixed text with mostly English should be 'en'."""
        text = "Hello world 你好"
        assert detect_language(text) == "en"

    def test_mixed_mostly_chinese(self):
        """Mixed text with mostly Chinese should be 'zh'."""
        text = "今天天气很好 hello"
        assert detect_language(text) == "zh"

    def test_empty_text(self):
        """Empty text should default to 'en'."""
        assert detect_language("") == "en"
