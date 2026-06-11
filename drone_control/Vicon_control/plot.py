#!/usr/bin/env python3
"""Static PNG overview of a Vicon-controller flight (thin launcher).

Reuses data_logging/plot_flight.py verbatim — just defaults discovery to
Vicon_control/flight_logs/. Writes flight_overview.png next to the flight.

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/plot.py                 # newest flight
  .venv/bin/python drone_control/Vicon_control/plot.py flight_logs/<stamp>/flight_synced.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(_REPO, "data_logging"))   # reuse plot_flight
import plot_flight  # noqa: E402

FLIGHT_LOGS = os.path.join(HERE, "flight_logs")


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not pos:
        syncs = plot_flight.list_synced(folder=FLIGHT_LOGS)
        if not syncs:
            sys.exit(f"No flight_synced.csv in {FLIGHT_LOGS}/ — run "
                     "Vicon_control/combine.py first.")
        flags = [a for a in sys.argv[1:] if a.startswith("-")]
        sys.argv = [sys.argv[0], syncs[0]] + flags
    plot_flight.main()


if __name__ == "__main__":
    main()
