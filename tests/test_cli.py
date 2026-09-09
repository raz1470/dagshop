"""Tests for `cli.py`: argument parsing and the `main()` wiring around
`server.create_app` and `uvicorn.run`.

`create_app` and `uvicorn.run` are mocked throughout (both are already
covered on their own terms: `create_app` in `test_server.py`, `uvicorn`
by its own upstream test suite) so these tests exercise only cli.py's
own logic: flag parsing, the file-existence and port-availability
preflight checks, and the create_app/uvicorn call wiring. No test in
this file actually starts a live server.
"""

from __future__ import annotations

import socket
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dagshop import cli
from dagshop.graph import DAGModel


def _write_csv(tmp_path, name: str = "data.csv"):
    path = tmp_path / name
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}).to_csv(path, index=False)
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until(condition, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


@pytest.fixture(autouse=True)
def _fast_browser_delay(monkeypatch):
    # Real launches wait _BROWSER_OPEN_DELAY_SECONDS before opening a tab
    # (see cli.py's docstring for why). Zero it out so browser-opening
    # tests don't spend real wall-clock time on it.
    monkeypatch.setattr(cli, "_BROWSER_OPEN_DELAY_SECONDS", 0.0)


@pytest.fixture
def mock_create_app(monkeypatch):
    mock = MagicMock(return_value=MagicMock(name="app"))
    monkeypatch.setattr(cli, "create_app", mock)
    return mock


@pytest.fixture
def mock_uvicorn_run(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli.uvicorn, "run", mock)
    return mock


# -- argument parsing ---------------------------------------------------------


def test_parser_requires_launch_subcommand():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults(tmp_path):
    parser = cli._build_parser()
    args = parser.parse_args(["launch", str(tmp_path / "data.csv")])
    assert args.treatments is None
    assert args.outcomes is None
    assert args.max_rows == 5000
    assert args.test_size == 0.2
    assert args.random_state == 0
    assert args.plot_grid_size == 50
    assert args.session is None
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.no_browser is False


def test_parser_repeatable_treatment_and_outcome_flags(tmp_path):
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "launch",
            str(tmp_path / "data.csv"),
            "--treatment",
            "x",
            "--treatment",
            "y",
            "--outcome",
            "z",
        ]
    )
    assert args.treatments == ["x", "y"]
    assert args.outcomes == ["z"]


def test_parser_overrides(tmp_path):
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "launch",
            str(tmp_path / "data.csv"),
            "--max-rows",
            "100",
            "--test-size",
            "0.3",
            "--random-state",
            "7",
            "--plot-grid-size",
            "20",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--no-browser",
        ]
    )
    assert args.max_rows == 100
    assert args.test_size == pytest.approx(0.3)
    assert args.random_state == 7
    assert args.plot_grid_size == 20
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.no_browser is True


# -- main(): file existence preflight -----------------------------------------


def test_main_rejects_missing_data_file(tmp_path, capsys, mock_create_app):
    missing = tmp_path / "nope.csv"
    with pytest.raises(SystemExit):
        cli.main(["launch", str(missing)])
    assert "data file not found" in capsys.readouterr().err
    mock_create_app.assert_not_called()


def test_main_rejects_missing_session_file(tmp_path, capsys, mock_create_app):
    data = _write_csv(tmp_path)
    missing_session = tmp_path / "nope.json"
    with pytest.raises(SystemExit):
        cli.main(["launch", str(data), "--session", str(missing_session)])
    assert "session file not found" in capsys.readouterr().err
    mock_create_app.assert_not_called()


# -- main(): create_app wiring -------------------------------------------------


def test_main_calls_create_app_with_parsed_args(mock_create_app, mock_uvicorn_run, tmp_path):
    data = _write_csv(tmp_path)
    port = _free_port()
    cli.main(
        [
            "launch",
            str(data),
            "--treatment",
            "a",
            "--outcome",
            "b",
            "--max-rows",
            "100",
            "--test-size",
            "0.25",
            "--random-state",
            "3",
            "--plot-grid-size",
            "10",
            "--port",
            str(port),
            "--no-browser",
        ]
    )
    mock_create_app.assert_called_once_with(
        data,
        treatments=["a"],
        outcomes=["b"],
        max_rows=100,
        test_size=0.25,
        random_state=3,
        plot_grid_size=10,
        initial_session=None,
    )
    mock_uvicorn_run.assert_called_once_with(
        mock_create_app.return_value, host="127.0.0.1", port=port
    )


def test_main_passes_session_path_through(mock_create_app, mock_uvicorn_run, tmp_path):
    data = _write_csv(tmp_path)
    session_path = tmp_path / "session.json"
    DAGModel().save_session(session_path)
    port = _free_port()
    cli.main(["launch", str(data), "--session", str(session_path), "--port", str(port)])
    assert mock_create_app.call_args.kwargs["initial_session"] == session_path


def test_main_surfaces_create_app_value_error(mock_create_app, capsys, tmp_path):
    data = _write_csv(tmp_path)
    mock_create_app.side_effect = ValueError("columns cannot be both a treatment and an outcome")
    with pytest.raises(SystemExit):
        cli.main(["launch", str(data), "--treatment", "a", "--outcome", "a"])
    assert "cannot be both a treatment and an outcome" in capsys.readouterr().err


# -- main(): port preflight ----------------------------------------------------


def test_main_rejects_port_already_in_use(mock_create_app, capsys, tmp_path):
    data = _write_csv(tmp_path)
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen(1)
        with pytest.raises(SystemExit):
            cli.main(["launch", str(data), "--port", str(port)])
    assert "already in use" in capsys.readouterr().err


def test_check_port_available_raises_on_bound_port():
    parser = cli._build_parser()
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen(1)
        with pytest.raises(SystemExit):
            cli._check_port_available(parser, "127.0.0.1", port)


# -- main(): browser auto-open --------------------------------------------------


def test_main_opens_browser_by_default(mock_create_app, mock_uvicorn_run, monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    data = _write_csv(tmp_path)
    port = _free_port()
    cli.main(["launch", str(data), "--host", "127.0.0.1", "--port", str(port)])
    assert _wait_until(lambda: opened == [f"http://127.0.0.1:{port}/"])


def test_main_skips_browser_with_no_browser_flag(
    mock_create_app, mock_uvicorn_run, monkeypatch, tmp_path
):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    data = _write_csv(tmp_path)
    port = _free_port()
    cli.main(["launch", str(data), "--port", str(port), "--no-browser"])
    # give a would-be browser-opening thread a moment to (wrongly) fire
    time.sleep(0.05)
    assert opened == []


def test_open_browser_after_delay_uses_explicit_delay(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    cli._open_browser_after_delay("http://127.0.0.1:8000/", delay=0.0)
    assert opened == ["http://127.0.0.1:8000/"]
