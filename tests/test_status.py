from pipeline.status import normalize_status, is_finished, is_upcoming


def test_normalize_finished():
    assert normalize_status("finished") == "finished"
    assert normalize_status("FT") == "finished"


def test_normalize_upcoming():
    assert normalize_status("notstarted") == "upcoming"
    assert normalize_status("scheduled") == "upcoming"


def test_normalize_cancelled():
    assert normalize_status("postponed") == "cancelled"


def test_is_finished():
    assert is_finished("finished") is True
    assert is_finished("notstarted") is False
