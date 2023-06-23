from os import path as op

from setuptools import find_packages, setup


def _read(f):
    return open(op.join(op.dirname(__file__), f)).read() if op.exists(f) else ""


install_requires = [
    l for l in _read("requirements.txt").split("\n") if l and not l.startswith("#") and not l.startswith("-")
]

setup(
    name="robot_io",
    version="0.0.1",
    packages=find_packages(exclude=["misc"]),
    install_requires=install_requires,
)
