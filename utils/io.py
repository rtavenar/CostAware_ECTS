import sys


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def tee(fp, s):
    fp.write(s)
    sys.stdout.write(s)


def teeln(fp, s):
    tee(fp, s + "\n")