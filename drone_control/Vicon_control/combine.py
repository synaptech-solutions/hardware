#!/usr/bin/env python3
"""Merge a Vicon-controller flight → flight_synced.csv (thin launcher).

Identical to data_logging/combine_flight.py — it just defaults the session to the
newest one under Vicon_control/flight_logs/ instead of data_logging/recordings/.
Drop the flight's .bbl into the session's blackbox/ subfolder first.

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/combine.py                 # newest flight
  .venv/bin/python drone_control/Vicon_control/combine.py flight_logs/<stamp>
  .venv/bin/python drone_control/Vicon_control/combine.py <session> --offset 0.015
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(_REPO, "data_logging"))   # reuse combine_flight
import combine_flight  # noqa: E402

FLIGHT_LOGS = os.path.join(HERE, "flight_logs")


def newest_session():
    dirs = [d for d in glob.glob(os.path.join(FLIGHT_LOGS, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "session.json"))]
    if not dirs:
        sys.exit(f"No Vicon flight session in {FLIGHT_LOGS}/ — fly + record first.")
    return max(dirs, key=os.path.getmtime)


def main():
    # Default the positional session to the newest flight_logs/ flight; pass any
    # explicit path / flags straight through to combine_flight.main().
    rest = sys.argv[1:]
    if not rest or rest[0].startswith("-"):
        sys.argv = [sys.argv[0], newest_session()] + rest
    combine_flight.main()


if __name__ == "__main__":
    main()
