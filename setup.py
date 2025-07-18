from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1'
DESCRIPTION = 'Empirical Industrial Organization'

# Setting up
setup(
    name="empio",
    version=VERSION,
    author="Gustavo Amarante",
    maintainer="Gustavo Amarante",
    maintainer_email="developer@dsgepy.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'seaborn',
        'statsmodels',
        'tqdm',
        'xlogit',
    ],
    keywords=[
        'empirical industrial organization',
        'industrial organization',
    ],
)
