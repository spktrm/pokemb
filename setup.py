from setuptools import find_packages, setup

from pokemb import __version__

requirements = ["numpy", "pandas", "scikit_learn", "torch", "sentence-transformers"]

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
