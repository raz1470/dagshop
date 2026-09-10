"""`dagshop` command-line entrypoints and internal PyPI packaging.

Last module in the build order (SCOPE.md build order step 4). Two
subcommands:

- `dagshop launch data.csv` -- wraps `server.py`'s `create_app` (the
  entire boundary this module owns, per `server.py`'s own module
  docstring) with argument parsing and a uvicorn run loop; no
  application logic lives here.
- `dagshop generate-demo-data OUTPUT.csv` -- wraps `demo_data.py`'s
  `make_demo_data`/`make_csat_demo_data` (SCOPE.md's "Manual testing"
  section), for someone without a real dataset yet to generate one with
  a known causal structure and try `launch` against it. Added session 7,
  after Ryan asked for a way to test the app without his own data.
  `--scenario` (session 12) picks which generator: `confounder`
  (default, the original) or `csat` (the causal attribution feature's
  multi-hop demo, SCOPE.md build order step 3).

Decisions from NOTES.md session 6 (the `launch` subcommand), all asked
of and confirmed by Ryan before writing this module:

- **CLI framework: stdlib `argparse`.** Zero new dependency, so no new
  entry on SCOPE.md's dependency-telemetry-audit list.
- **Treatment/outcome flags are repeatable**, not comma-separated:
  `dagshop launch data.csv --treatment X --treatment Y --outcome Z`.
  Standard `argparse` `action="append"` pattern; no ambiguity if a
  column name itself contains a comma.
- **Internal registry: not set up yet.** SCOPE.md says "internal
  PyPI/package registry" but names no concrete target (private PyPI
  server, AWS CodeArtifact, GitHub Packages, ...) and none exists today.
  This slice adds `[project.scripts]` and confirms `uv build` produces a
  correct wheel/sdist; it does not add a publish workflow. Revisit once
  a registry is chosen (see NOTES.md session 6 and SCOPE.md open items).
- **`--host`/`--port` flags, browser auto-open, and `--session` resume**
  are all in scope for the `launch` subcommand (Ryan picked all three
  when asked which launch-time behaviors to add).

Decision from NOTES.md session 7 (the `generate-demo-data` subcommand,
asked of and confirmed by Ryan before writing it): ships as a real
subcommand rather than a standalone dev script under `scripts/`, so
anyone who `pip install`s dagshop without their own data yet can try
the tool immediately. This does add a small, permanent, user-facing
surface for what's really a testing convenience -- the trade-off Ryan
picked over keeping it dev-only.

Not built here: the association-scan defaults duplicated in `launch`
below (`max_rows`, `test_size`, `random_state`, `plot_grid_size`)
intentionally mirror `server.create_app`'s own defaults exactly, so
`dagshop launch data.csv` with no flags behaves identically to calling
`create_app` with no keyword overrides. If those defaults ever change
in `server.py`, change them here too.
"""

from __future__ import annotations

import argparse
import shlex
import socket
import threading
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from dagshop.demo_data import make_csat_demo_data, make_demo_data
from dagshop.server import create_app

_BROWSER_OPEN_DELAY_SECONDS = 1.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dagshop",
        description=("Interactive DAG-drawing workshop tool for PM and data scientist pairs."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_launch_subparser(subparsers)
    _add_generate_demo_data_subparser(subparsers)
    return parser


def _add_launch_subparser(subparsers: argparse._SubParsersAction) -> None:
    launch = subparsers.add_parser(
        "launch",
        help="Launch the DAGshop workshop UI for a CSV dataset.",
    )
    launch.add_argument("data", type=Path, help="Path to the input CSV file.")
    launch.add_argument(
        "--treatment",
        dest="treatments",
        action="append",
        default=None,
        metavar="COLUMN",
        help="Designate COLUMN as a treatment variable. Repeat for more than one.",
    )
    launch.add_argument(
        "--outcome",
        dest="outcomes",
        action="append",
        default=None,
        metavar="COLUMN",
        help="Designate COLUMN as an outcome variable. Repeat for more than one.",
    )
    launch.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        metavar="N",
        help="Subsample size for the association scan (default: 5000).",
    )
    launch.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRACTION",
        help="Held-out fraction per pair for scoring (default: 0.2).",
    )
    launch.add_argument(
        "--random-state",
        type=int,
        default=0,
        metavar="SEED",
        help="Random seed for subsampling, splits, and initial node layout (default: 0).",
    )
    launch.add_argument(
        "--plot-grid-size",
        type=int,
        default=50,
        metavar="N",
        help="Points in each pairwise partial-dependence curve (default: 50).",
    )
    launch.add_argument(
        "--session",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Resume a previously saved session file instead of starting from a "
            "fresh layout. The association scan still runs against `data` either way."
        ),
    )
    launch.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1).",
    )
    launch.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000).",
    )
    launch.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser tab once the server is up.",
    )


def _add_generate_demo_data_subparser(subparsers: argparse._SubParsersAction) -> None:
    demo = subparsers.add_parser(
        "generate-demo-data",
        help=(
            "Write a synthetic CSV with a known causal structure, for trying "
            "`launch` without a real dataset."
        ),
    )
    demo.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("inputs/demo.csv"),
        help="Path to write the generated CSV to (default: inputs/demo.csv).",
    )
    demo.add_argument(
        "--scenario",
        choices=["confounder", "csat"],
        default="confounder",
        help=(
            "Which synthetic scenario to generate (default: confounder). "
            "confounder: the original small confounder/treatment/mediator/outcome "
            "DAG. csat: the multi-hop customer-service operations DAG for the "
            "causal attribution feature -- see dagshop.demo_data's module docstring "
            "for both."
        ),
    )
    demo.add_argument(
        "--n-rows",
        type=int,
        default=500,
        metavar="N",
        help="Number of rows to generate (default: 500).",
    )
    demo.add_argument(
        "--random-state",
        type=int,
        default=0,
        metavar="SEED",
        help="Random seed (default: 0).",
    )
    demo.add_argument(
        "--force",
        action="store_true",
        help="Overwrite OUTPUT if it already exists.",
    )


def _check_port_available(parser: argparse.ArgumentParser, host: str, port: int) -> None:
    """Fail fast with a clear message if `host:port` is already bound.

    A bind failure inside uvicorn's own startup surfaces as a logged
    error and a bare process exit from deep in its async startup
    sequence, not a catchable exception with a useful message at this
    layer. Doing the bind check here first, with `SO_REUSEADDR` so it
    doesn't trip on a socket still in TIME_WAIT, gives a specific,
    actionable error instead. This is a preflight check, not a lock: a
    narrow race remains if something else binds the port between this
    check and uvicorn's own bind a moment later.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            parser.error(
                f"{host}:{port} is already in use ({exc}). "
                "Pick a different port with --port, or stop whatever is using it."
            )


def _open_browser_after_delay(url: str, delay: float | None = None) -> None:
    """Open `url` in a browser tab shortly after `uvicorn.run` is called.

    A short fixed delay rather than polling uvicorn's internal readiness
    state (judgment call, not asked -- flagging per PREFERENCES.md):
    `create_app` has already finished (the association scan, the
    typically-slow part, runs before this function is ever scheduled),
    so by the time `main` calls `uvicorn.run`, binding to a local
    host/port is near-instant. Reaching into uvicorn's `Config`/`Server`
    internals just to observe a `.started` flag seemed like more
    machinery than a one-second wait buys here.

    `delay` defaults to the module-level `_BROWSER_OPEN_DELAY_SECONDS`
    read at call time, not bound as the parameter's default value, so
    tests can monkeypatch that module attribute to skip the real wait
    without needing to pass `delay` through every layer of `main`.
    """
    if delay is None:
        delay = _BROWSER_OPEN_DELAY_SECONDS
    time.sleep(delay)
    webbrowser.open(url)


def _run_launch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.data.exists():
        parser.error(f"data file not found: {args.data}")
    if args.session is not None and not args.session.exists():
        parser.error(f"session file not found: {args.session}")

    try:
        app = create_app(
            args.data,
            treatments=args.treatments,
            outcomes=args.outcomes,
            max_rows=args.max_rows,
            test_size=args.test_size,
            random_state=args.random_state,
            plot_grid_size=args.plot_grid_size,
            initial_session=args.session,
        )
    except ValueError as exc:
        parser.error(str(exc))

    _check_port_available(parser, args.host, args.port)

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Thread(target=_open_browser_after_delay, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)


def _run_generate_demo_data(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.output.exists() and not args.force:
        parser.error(
            f"{args.output} already exists. Pass --force to overwrite, "
            "or choose a different output path."
        )

    if args.scenario == "csat":
        data = make_csat_demo_data(n_rows=args.n_rows, random_state=args.random_state)
        data.to_csv(args.output, index=False)

        print(f"Wrote {len(data)} rows to {args.output}")
        print()
        print("Ground truth (see dagshop.demo_data's module docstring for exact coefficients):")
        print("  roots:        age, friction_severity, time_to_respond")
        print("  operational:  num_transfers, num_escalations, num_agents_spoken_to")
        print("  resolution:   time_to_resolve, resolved (binary), repeat_contact (binary)")
        print("  target:       csat")
        print()
        print("Try:")
        print(f"  dagshop launch {shlex.quote(str(args.output))} --outcome csat")
        return

    data = make_demo_data(n_rows=args.n_rows, random_state=args.random_state)
    data.to_csv(args.output, index=False)

    print(f"Wrote {len(data)} rows to {args.output}")
    print()
    print("Ground truth (see dagshop.demo_data's module docstring for exact coefficients):")
    print("  confounders:  age, prior_engagement")
    print("  treatment:    treatment (binary)")
    print("  mediator:     mediator")
    print("  outcome:      outcome")
    print("  noise:        unrelated_score, unrelated_flag (should rank low either table)")
    print()
    print("Try:")
    print(
        f"  dagshop launch {shlex.quote(str(args.output))} --treatment treatment --outcome outcome"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "launch":
        _run_launch(args, parser)
    else:
        # argparse's subparsers (dest="command", required=True) only
        # ever hand back "launch" or "generate-demo-data", so this is
        # the generate-demo-data branch, not an unchecked fallback.
        _run_generate_demo_data(args, parser)


if __name__ == "__main__":
    main()
