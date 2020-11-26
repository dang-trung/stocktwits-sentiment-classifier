"""Setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib
from setuptools import setup, find_packages

wd = pathlib.Path(__file__).parent.resolve()
long_description = (wd / 'README.md').read_text(encoding='utf-8')
with open(wd / 'requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='sent-classifier',
    version='0.0.0',
    description='Classify sentiments of StockTwits crypto-related messages',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dang-trung/stocktwits-sentiment-classifier',
    author='Trung Dang',
    author_email='dangtrung96@gmail.com',
    packages=find_packages(include=['sentiment_classifier']),
    package_data={'external': ['data/00_external/*.csv']},
    python_requires='>=3.5, <4',
    install_requires=requirements,
)
