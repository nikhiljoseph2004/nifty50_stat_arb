"""
Setup configuration for nifty50_stat_arb package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nifty50-stat-arb",
    version="0.1.0",
    author="Nikhil Joseph",
    author_email="",
    description="A pairs trading strategy for Nifty 50 stocks based on statistical arbitrage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikhiljoseph2004/nifty50_stat_arb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nifty50-stat-arb=main:main",
        ],
    },
)
