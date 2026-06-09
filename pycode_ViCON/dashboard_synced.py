"""Interactive dashboard for sync_log.py output (*_synced.mat in DataExchange/).

Thin launcher over the shared flight_dashboard module (so this and
data_logging/dashboard_flight.py have identical UI + functionality). Opens a
Plotly Dash web app in your browser — nothing written to disk (use
plot_synced.py for static PNGs). File discovery reuses plot_synced.py.

Usage (run with the repo venv, ../.venv):
  ../.venv/bin/python dashboard_synced.py                       # newest *_synced.mat
  ../.venv/bin/python dashboard_synced.py --pick                # menu
  ../.venv/bin/python dashboard_synced.py DataExchange/foo_synced.mat
  ../.venv/bin/python dashboard_synced.py --port 8060 --no-browser
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_synced import list_synced, select_synced
import flight_dashboard

DATAEXCHANGE = os.path.join(HERE, "DataExchange")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="*_synced.mat (default: newest in DataExchange/)")
    ap.add_argument("--pick", action="store_true", help="choose from a menu")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    if args.file:
        path = args.file
    elif args.pick:
        path = select_synced(DATAEXCHANGE)
    else:
        path = list_synced(DATAEXCHANGE)[0]

    flight_dashboard.serve(path, "Synced-log dashboard", args.port, not args.no_browser)


if __name__ == "__main__":
    main()
