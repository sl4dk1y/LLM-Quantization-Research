"""Базовые метрики поверх нормализованных task outputs.

Адаптеры задач должны извлечь prediction/reference по официальным правилам.
Этот модуль не заменяет official scorer и не определяет формат датасетов.
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Iterable

SCORER_VERSION = "quantbench-basic-metrics-0.1"

def normalize_answer(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prediction и reference должны быть строками")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))

def token_f1(prediction: str, reference: str) -> float:
    pred = re.findall(r"\w+", normalize_answer(prediction), flags=re.UNICODE)
    ref = re.findall(r"\w+", normalize_answer(reference), flags=re.UNICODE)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def accuracy(pairs: Iterable[tuple[str, str]]) -> float:
    values = [exact_match(pred, ref) for pred, ref in pairs]
    if not values:
        raise ValueError("Нельзя вычислить Accuracy на пустом наборе")
    return sum(values) / len(values)
