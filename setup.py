from setuptools import find_packages, setup

from pokemb import __version__

requirements = ["numpy", "torch"]

generate_requirements = ["pandas", "scikit_learn", "sentence-transformers"]

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
    extras_require=dict(generate=generate_requirements),
)
