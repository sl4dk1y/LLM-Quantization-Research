import json

import pytest

from quantbench.dataset_pipeline import (
    LIBRA_REVISION, LIBRA_TEMPLATES, MERA_REVISION, LibraScorer, OfficialScorerUnavailable,
    export_split_ids, extract_unambiguous_label, postprocess_mera_prediction,
    prepare_export, score_mera,
)


class _Parsed:
    def __init__(self, word):
        self.normal_form = word


class _Analyzer:
    def parse(self, word):
        return [_Parsed(word)]


def _mera(id_, *, split=None, output="B"):
    row = {"instruction": "Задача: {text}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}",
           "inputs": {"text": "вопрос", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d"},
           "outputs": output, "meta": {"id": id_}}
    if split:
        row["split"] = split
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _prepare(**kwargs):
    kwargs["source_provenance"] = {"fixture": True, "local_path": str(kwargs["source_path"])}
    kwargs["allow_test_fixture"] = True
    return prepare_export(**kwargs)


def test_mera_export_is_deterministic_and_preserves_source_labels(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_mera(i) for i in range(7)])
    kwargs = dict(source_path=source, dataset="MERA", task="MathLogicQA", split="train",
                  revision=MERA_REVISION, count=4)
    first = _prepare(**kwargs, output_path=tmp_path / "a.jsonl", manifest_path=tmp_path / "a.json")
    second = _prepare(**kwargs, output_path=tmp_path / "b.jsonl", manifest_path=tmp_path / "b.json")
    assert first["item_ids"] == second["item_ids"]
    assert first["prepared_sha256"] == second["prepared_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert all(row["source_label"] == "B" and row["references"] == ["B"] for row in rows)
    assert all(len(row["prompt_sha256"]) == 64 and len(row["prompt_template_sha256"]) == 64 for row in rows)
    assert first["source_provenance"]["verified"] is False


def test_mera_export_fails_closed_on_missing_labels_split_and_duplicate_ids(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [{**_mera(1), "outputs": None}])
    with pytest.raises(ValueError, match="outputs"):
        _prepare(source_path=source, output_path=tmp_path / "o", manifest_path=tmp_path / "m",
                 dataset="MERA", task="MathLogicQA", split="train", revision=MERA_REVISION, count=1)
    _write(source, [_mera(1, split="test")])
    with pytest.raises(ValueError, match="split"):
        _prepare(source_path=source, output_path=tmp_path / "o", manifest_path=tmp_path / "m",
                 dataset="MERA", task="MathLogicQA", split="train", revision=MERA_REVISION, count=1)
    _write(source, [_mera(1), _mera(1)])
    with pytest.raises(ValueError, match="Повторяющийся"):
        _prepare(source_path=source, output_path=tmp_path / "o", manifest_path=tmp_path / "m",
                 dataset="MERA", task="MathLogicQA", split="train", revision=MERA_REVISION, count=1)


def test_heldout_export_requires_disjoint_ids_and_rejects_overlap(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_mera(5)])
    with pytest.raises(ValueError, match="disjoint-source"):
        _prepare(source_path=source, output_path=tmp_path / "a", manifest_path=tmp_path / "am",
                 dataset="MERA", task="MathLogicQA", split="test", revision=MERA_REVISION, count=1)
    with pytest.raises(ValueError, match="пересечение"):
        _prepare(source_path=source, output_path=tmp_path / "b", manifest_path=tmp_path / "bm",
                 dataset="MERA", task="MathLogicQA", split="test", revision=MERA_REVISION,
                 count=1, disjoint_ids=["5"])


def test_source_provenance_is_mandatory_even_with_a_pinned_revision(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_mera("item")])
    with pytest.raises(ValueError, match="source provenance"):
        prepare_export(source_path=source, output_path=tmp_path / "o", manifest_path=tmp_path / "m",
                       dataset="MERA", task="MathLogicQA", split="train", revision=MERA_REVISION, count=1)


def test_split_id_manifest_is_deterministic_and_contains_no_answers(tmp_path):
    source = tmp_path / "train.jsonl"
    _write(source, [_mera(1), _mera(2)])
    manifest = export_split_ids(source_path=source, output_path=tmp_path / "ids.json", dataset="MERA",
                                task="MathLogicQA", split="train", revision=MERA_REVISION)
    assert manifest["item_ids"] == ["1", "2"]
    assert "outputs" not in (tmp_path / "ids.json").read_text()


def test_export_refuses_source_alias_and_changed_output_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_mera(1)])
    kwargs = dict(source_path=source, dataset="MERA", task="MathLogicQA", split="train",
                  revision=MERA_REVISION, count=1)
    with pytest.raises(ValueError, match="должны быть разными"):
        _prepare(**kwargs, output_path=source, manifest_path=tmp_path / "m.json")
    output = tmp_path / "out.jsonl"
    _prepare(**kwargs, output_path=output, manifest_path=tmp_path / "m.json")
    output.write_text("foreign content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="перезаписи"):
        _prepare(**kwargs, output_path=output, manifest_path=tmp_path / "m.json")


@pytest.mark.parametrize("task, phrase", [
    ("ru_babilong_qa3", "фактами о местоположении"),
    ("long_context_multiq", "длинный текст"),
    ("matreshka_names", "несколько диалогов"),
])
def test_libra_templates_filter_mixed_length_rows_and_preserve_references(tmp_path, task, phrase):
    source = tmp_path / "libra.jsonl"
    rows = [
        {"id": "x1", "length": "8k", "context": "Другой контекст", "input": "Кто?",
         "positive_outputs": ["Александр", "Саша"]},
        {"id": "x2", "length": "4k", "context": "Полный контекст", "input": "Вопрос?",
         "positive_outputs": ["Иван", "Иван Петров"], "negative_outputs": ["Петр"]},
        {"id": "x3", "length": "16k", "context": "Третий", "input": "Кто?",
         "positive_outputs": ["Мария"]},
    ]
    _write(source, rows)
    output = tmp_path / f"{task}.jsonl"
    manifest = _prepare(source_path=source, output_path=output, manifest_path=tmp_path / f"{task}.json",
                        dataset="LIBRA", task=task, split="test", revision=LIBRA_REVISION,
                        count=1, context_length="4K")
    exported = json.loads(output.read_text())
    assert exported["id"] == "x2"
    assert exported["references"] == rows[1]["positive_outputs"]
    assert exported["source_record"]["context"] == "Полный контекст"
    assert phrase in exported["prompt"]
    assert "Другой контекст" not in exported["prompt"]
    assert manifest["sample_count"] == 1


def test_libra_length_filter_fails_closed_when_requested_bin_is_empty(tmp_path):
    source = tmp_path / "libra.jsonl"
    _write(source, [{"id": "x", "length": "8k", "context": "Контекст", "input": "Вопрос",
                    "positive_outputs": ["Ответ"]}])
    with pytest.raises(ValueError, match="Запрошено 1 элементов, доступно 0"):
        _prepare(source_path=source, output_path=tmp_path / "o", manifest_path=tmp_path / "m",
                 dataset="LIBRA", task="ru_babilong_qa3", split="test", revision=LIBRA_REVISION,
                 count=1, context_length="4K")


def test_libra_scorer_uses_any_reference_and_official_normalization_shape():
    scorer = LibraScorer(analyzer=_Analyzer())
    assert scorer.score_item("Ответ: Иван Петров", ["Пётр", "Иван Петров"]) == 1.0
    assert scorer.score_item("Мария", ["Иван", "Пётр"]) == 0.0
    assert scorer.aggregate([{"prediction": "Иван", "references": ["Иван"], "context_length_class": "4K"}]) == {"4K": 1.0}


def test_libra_templates_match_pinned_official_prompt_text():
    assert LIBRA_TEMPLATES["ru_babilong_qa3"] == (
        "Я даю тебе контекст с фактами о местоположении и действиях разных людей. Тебе нужно ответить на вопрос, "
        "основываясь только на фактической информации. Если человек взял предмет в одном месте и отправился в другое "
        "место, этот предмет также находится во втором месте. Если человек оставил предмет в первом метсе и перешел "
        "во второе место, предмет остается в первом месте.\n{{context}}\nОтветь на вопрос как можно более кратко.\n"
        "Вопрос: {{input}}\nОтвет:"
    )
    assert LIBRA_TEMPLATES["long_context_multiq"] == (
        "Тебе предоставляется длинный текст, в котором нужно найти ответ на вопрос.\n{{context}}\n"
        "Найди ответ в тексте на следующий вопрос.\nВопрос:{{input}}\nОтвет:"
    )
    assert LIBRA_TEMPLATES["matreshka_names"] == (
        "Тебе предоставляются несколько диалогов. Запомни имена людей и темы, на которые они говорили.\n"
        "{{context}}\nВ ответе укажи только имя собеседника, который говорил на тему из следующего вопроса.\n"
        "Вопрос:{{input}}\nОтвет:"
    )


def test_libra_real_russian_lemmatizer_matches_inflected_reference():
    scorer = LibraScorer()
    assert scorer.score_item("Коты играют на столах", ["кот"]) == 1.0


def test_libra_scorer_requires_pymorphy3(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_pymorphy(name, *args, **kwargs):
        if name == "pymorphy3":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_pymorphy)
    with pytest.raises(OfficialScorerUnavailable, match="pymorphy3"):
        LibraScorer()


def test_official_mera_choice_postprocessing_and_metrics():
    assert postprocess_mera_prediction("MathLogicQA", "Ответ: B\nДумаю, возможно C") == "B"
    assert postprocess_mera_prediction("MathLogicQA", "Ответ: E") == "-1"
    assert postprocess_mera_prediction("PARus", "Вариант 2\n3") == "2"
    score = score_mera(task="MathLogicQA", predictions=["A", "Ответ: B", "E"],
                       references=["A", "A", "C"])
    assert score["metrics"]["exact_match"] == pytest.approx(1 / 3)
    assert score["metrics"]["accuracy_equivalent"] == pytest.approx(1 / 3)
    assert score["predictions_processed"] == ["A", "B", "-1"]


def test_official_multiq_squad_metrics_and_multiple_answers():
    score = score_mera(task="MultiQ", predictions=[" Москва. ", "Бонн", ""],
                       references=["Москва;Столица", "Париж;Берлин", "неизвестно"])
    assert score["metrics"]["em"] == pytest.approx(1 / 3)
    assert score["metrics"]["f1"] == pytest.approx(1 / 3)


def test_official_ruopenbook_macro_f1_and_parus_accuracy():
    score = score_mera(task="ruOpenBookQA", predictions=["Ответ A", "B", "нет"],
                       references=["A", "A", "C"])
    assert score["metrics"]["exact_match"] == pytest.approx(1 / 3)
    assert "f1" in score["metrics"]
    parus = score_mera(task="PARus", predictions=["1", "2"], references=["1", "1"])
    assert parus["metrics"]["exact_match"] == pytest.approx(0.5)


def test_choice_diagnostic_does_not_replace_official_extraction():
    assert extract_unambiguous_label("Ответ: B", ["A", "B", "C", "D"]) == "B"
    assert extract_unambiguous_label("A или B", ["A", "B", "C", "D"]) is None
    assert extract_unambiguous_label("нет варианта", ["A", "B", "C", "D"]) is None
