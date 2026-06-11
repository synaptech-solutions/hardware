#!/usr/bin/env python3
"""Interactive web dashboard for a Vicon-controller flight (thin launcher).

Reuses the SAME shared module (pycode_ViCON/flight_dashboard.py) as
data_logging/dashboard_flight.py — identical UI — just defaulting discovery to
Vicon_control/flight_logs/. Pose + commands + telemetry + FC blackbox + video,
exactly like reviewing a hand-flown flight.

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/dashboard.py               # newest flight
  .venv/bin/python drone_control/Vicon_control/dashboard.py --pick        # menu
  .venv/bin/python drone_control/Vicon_control/dashboard.py flight_logs/<stamp>/flight_synced.csv
  .venv/bin/python drone_control/Vicon_control/dashboard.py --port 8060 --no-browser
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(_REPO, "data_logging"))     # for plot_flight discovery
sys.path.insert(0, os.path.join(_REPO, "pycode_ViCON"))     # for flight_dashboard

from plot_flight import list_synced, select_synced  # noqa: E402
import flight_dashboard  # noqa: E402

FLIGHT_LOGS = os.path.join(HERE, "flight_logs")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="flight_synced.csv (default: newest in flight_logs/)")
    ap.add_argument("--pick", action="store_true", help="choose from a menu")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    if args.file:
        path = args.file
    elif args.pick:
        path = select_synced(folder=FLIGHT_LOGS)
    else:
        syncs = list_synced(folder=FLIGHT_LOGS)
        if not syncs:
            sys.exit(f"No flight_synced.csv in {FLIGHT_LOGS}/ — run "
                     "Vicon_control/combine.py first.")
        path = syncs[0]

    flight_dashboard.serve(path, "Vicon hover dashboard", args.port, not args.no_browser)


if __name__ == "__main__":
    main()
