import logging
import re
import time
from typing import Any

logger = logging.getLogger("preprocessing")

# 中文常见标点符号
CJK_PUNCTUATION = set("，。！？；：、""''（）《》【】…—～·")
END_MARKS = set("。！？")


def split_text_to_lines(text: str, max_line_length: int = 2000) -> list[str]:
    """
    将文本按自然语言边界拆分成多行，处理语音转文字的长文本输入。

    策略：
    1. 优先按换行符拆分
    2. 按句末标点（。！？）拆分长行
    3. 极端长句按最大长度截断
    """
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result: list[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= max_line_length:
            result.append(stripped)
        else:
            # 按句末标点拆分
            sub_lines = _split_by_sentence_mark(stripped, max_line_length)
            result.extend(sub_lines)

    return result


def _split_by_sentence_mark(text: str, max_len: int) -> list[str]:
    """按句末标点拆分超长句子"""
    result: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in END_MARKS and len(current) >= 10:
            result.append(current.strip())
            current = ""
    if current.strip():
        result.append(current.strip())

    # 二次拆分仍然超长的行
    final: list[str] = []
    for line in result:
        if len(line) <= max_len:
            final.append(line)
        else:
            for i in range(0, len(line), max_len):
                final.append(line[i:i + max_len])
    return final


def clean_text(text: str) -> str:
    """
    基本文本清洗：
    - 去除多余空白
    - 统一全角/半角标点
    """
    text = text.strip()
    # 合并多个空白
    text = re.sub(r"[　 \t]+", " ", text)
    # 合并多个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def estimate_asr_noise_level(text: str) -> dict[str, Any]:
    """
    评估文本的语音转文字噪声水平。

    Returns:
        dict with keys:
            score (float): 0-1，越高越可能是 ASR 噪声文本
            indicators (list[str]): 判断指标
    """
    indicators: list[str] = []
    score = 0.0

    # 检查重复字词
    repeats = len(re.findall(r"(.)\1{2,}", text))
    if repeats > 2:
        score += 0.15
        indicators.append(f"字符连续重复({repeats}处)")

    # 检查缺少标点（长句无标点可能是ASR问题）
    no_punc_segments = re.findall(r"[^\n。，！？；：、]{30,}", text)
    if no_punc_segments:
        score += 0.2
        indicators.append(f"长段缺标点({len(no_punc_segments)}处)")

    # 检查碎片化短句
    short_lines = [l for l in text.split("\n") if 1 <= len(l.strip()) <= 5]
    if len(short_lines) > 3:
        score += 0.1
        indicators.append(f"碎片化短句({len(short_lines)}处)")

    # 检查不完整句子（以非标点结尾）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    incomplete = 0
    for line in lines:
        if line and line[-1] not in ".。！？\n":
            incomplete += 1
    if incomplete > len(lines) * 0.3:
        score += 0.15
        indicators.append(f"不完整句子({incomplete}/{len(lines)})")

    return {
        "score": round(min(1.0, score), 3),
        "indicators": indicators,
        "is_likely_noisy": score > 0.3,
    }
