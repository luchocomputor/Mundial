import pytest

from pipeline.placeholder import is_placeholder_team


@pytest.mark.parametrize(
    "team,expected",
    [
        ("Qatar", False),
        ("Ghana", False),
        ("Haiti", False),
        ("Germany", False),
        ("France", False),
        ("Winner Group A", True),
        ("Runner-up Group B", True),
        ("W1 Group C", True),
        ("3rd place Group D", True),
        ("Best 3rd", True),
        ("TBD", True),
        ("", True),
    ],
)
def test_placeholder_detection(team, expected):
    assert is_placeholder_team(team) == expected
