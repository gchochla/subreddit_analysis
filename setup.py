from setuptools import setup, find_packages

setup(
    name="subreddit_analysis",
    version="1.0.0",
    description="Analysis on Reddit with DDR",
    author="Georgios Chochlakis; Asaf Mazar",
    author_email="chochlak@usc.edu; amazar@usc.edu",
    packages=find_packages(),
    install_requires=[
        "praw==7.4.0",
        "ddr @ git+https://github.com/gchochla/DDR.git@master",
        "numpy==1.21.4",
        "scikit-learn==1.0.1",
        "pandas==1.3.4",
        "torch==1.10.0",
    ],
)
