from pathlib import Path
from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name='text2emotion',
    version='0.1',
    description='This package contains the code for the Text 2 emotions',
    author='Vineet Verma',
    author_email='vineetver@hotmail.com',
    url="https://github.com/vineetver/Text2Emotion",
    python_requires='==3.8',
    packages=find_namespace_packages(exclude=['test']),
    install_requires=[required_packages],
)
