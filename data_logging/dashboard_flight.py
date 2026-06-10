"""Interactive dashboard for combine_flight.py output (flight_synced.csv).

Thin launcher over the shared flight_dashboard module (so this and
pycode_ViCON/dashboard_synced.py have identical UI + functionality). Opens a
Plotly Dash web app in your browser — nothing written to disk (use
plot_flight.py for static PNGs). File discovery reuses plot_flight.py. Toggle on
the cmd_*/tlm_* panels to see the captured commands + telemetry alongside pose
and the FC blackbox. Legacy .mat sessions still open.

Usage (run with the repo venv):
  .venv/bin/python data_logging/dashboard_flight.py                 # newest flight
  .venv/bin/python data_logging/dashboard_flight.py --pick          # menu
  .venv/bin/python data_logging/dashboard_flight.py recordings/X/flight_synced.csv
  .venv/bin/python data_logging/dashboard_flight.py --port 8060 --no-browser
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # for plot_flight
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pycode_ViCON"))  # for flight_dashboard

from plot_flight import list_synced, select_synced
import flight_dashboard


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="flight_synced.csv (default: newest)")
    ap.add_argument("--pick", action="store_true", help="choose from a menu")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    if args.file:
        path = args.file
    elif args.pick:
        path = select_synced()
    else:
        path = list_synced()[0]

    flight_dashboard.serve(path, "Flight dashboard", args.port, not args.no_browser)


if __name__ == "__main__":
    main()
