"""pytest ortak ayarları: depo kökü import yoluna eklenir (from src import ...)."""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
