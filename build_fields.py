"""Put the downloaded products on the common grid (results/fields).

  main   ERA5, AMSR2, MASAM2 and their consensus; land mask; routes; georeferencing QA
  carra  CARRA-East inside its own domain (needs main)
  nic    NIC charts rasterised and assigned to every day (needs main; pyshp)

  python build_fields.py main nic --workers 12
"""
import argparse

from sicroute import carra, fields, nic
from sicroute.config import CFG


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("parts", nargs="*", choices=["main", "carra", "nic"], help="default: main nic")
    a.add_argument("--workers", type=int, default=8, help="processes for rasterising the NIC charts")
    a = a.parse_args()
    a.parts = a.parts or ["main", "nic"]
    if "main" in a.parts:
        fields.build_fields(CFG)
    if "carra" in a.parts:
        carra.build_fields(CFG)
    if "nic" in a.parts:
        nic.build_fields(CFG, a.workers)


if __name__ == "__main__":
    main()
