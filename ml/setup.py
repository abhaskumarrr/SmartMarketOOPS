#!/usr/bin/env python3
"""
Setup script for SMOOPs ML Module
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
readme = Path(__file__).parent / 'README.md'
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="smoops-ml",
    version="1.0.0",
    description="Smart Money Order Blocks Trading Bot ML Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abhas Kumar",
    author_email="109647440+abhaskumarrr@users.noreply.github.com",
    url="https://github.com/abhaskumarrr/SMOOPs_dev",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "smoops-ml=ml.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 