from setuptools import setup, find_packages

setup(
    name='text2emotion',
    version='0.1',
    description='This package contains the code for the Text 2 emotions',
    author='Vineet Verma',
    author_email='vineetver@hotmail.com',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'emoji',
        'tensorflow',
        'transformers',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'plotly',
        'seaborn',
        'pyspark',
        'jupyter'],
    entry_points={
        'console_scripts': [
            'clean_data = main.clean:main',
            'feature_engineering = main.feature_engineering:main',
            'train_model = main.train_model:main',
            'make_predictions = main.make_predictions:main',
        ],
    }
)
