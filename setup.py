#!/usr/bin/env python
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "src",
    version = "0.0.4",
    author = "Selle Bandstra",
    description = ("Simple python package to find stripy patterns in neurons"),
    license = "GNU",
    keywords = "example pytest tutorial sphinx",
    packages=['src', 'test'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
