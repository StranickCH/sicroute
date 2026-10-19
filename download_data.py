"""Download the input data into data/ (or $SICROUTE_DATA). Interrupted downloads resume.

  era5    ERA5 sea-ice cover, Copernicus CDS (needs a CDS account and ~/.cdsapirc)
  amsr2   AMSR2 ASI 6.25 km, University of Bremen (no account)
  masam2  MASAM2 v2, NSIDC (no account)
  carra   CARRA-East sea-ice fraction, Copernicus CDS (slow: the CDS queue may take hours)
  nic     U.S. National Ice Center weekly Arctic charts, NSIDC G10013 (no account)

  python download_data.py                  # era5 amsr2 masam2 nic
  python download_data.py carra
"""
import argparse

from sicroute import carra, fields, nic
from sicroute.config import CFG

SOURCES = {"era5": fields.download_era5, "amsr2": fields.download_amsr2, "masam2": fields.download_masam2,
           "carra": carra.download, "nic": nic.download}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("sources", nargs="*", choices=list(SOURCES), help="default: era5 amsr2 masam2 nic")
    for s in a.parse_args().sources or ["era5", "amsr2", "masam2", "nic"]:
        SOURCES[s](CFG)


if __name__ == "__main__":
    main()
