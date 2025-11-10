# bootstrap.py
import sys, os

def init():
    ROOT = os.path.dirname(__file__)
    if ROOT not in sys.path:
        sys.path.append(ROOT)