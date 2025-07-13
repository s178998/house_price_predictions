from setuptools import setup, find_packages

setup (
    name="myPipleline",
    version=0.1,
    packages=find_packages(),
    install_requires = [
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'kerastuner',
        'joblib',
    ],

    author='Ayodeji Osungbohun',
    description='A package for ML pipelines using TF and scikit-learn'
)


