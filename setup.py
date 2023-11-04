from setuptools import find_packages, setup

from pokemb import __version__

requirements = [
    "numpy==1.21.5",
    "pandas==2.0.2",
    "scikit_learn==1.3.2",
    "torch==1.13.0",
]

setup(
    name="pokemb",
    version=__version__,
    url="https://github.com/spktrm/pokemb",
    author="Joseph Twin",
    author_email="joseph.twin14@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=["wheel"],
    install_requires=requirements,
    package_data={"": ["*.ini"]},
)
