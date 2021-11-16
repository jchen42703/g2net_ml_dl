from setuptools import setup, find_packages

setup(
    name="g2net",
    version="0.0.1",
    description=
    "Automatically detecting gravitational waves in simulated signals",
    url="https://github.com/jchen42703/g2net_ml_dl",
    author=
    "Joseph Chen, Richard Chen, Takoda Denhof, Owen Hill, Julian Robinson",
    license="MIT License",
    packages=find_packages(),
    install_requires=[
        "torchsummary==1.5.1", "nnaudio==0.2.4", "timm==0.4.12",
        "matplotlib==3.3.2", "tensorboard>=2.4.0", "tqdm==4.50.2",
        "numpy==1.19.2", "seaborn==0.11.0", "packaging==20.4", "pandas==1.1.3",
        "scikit_learn>=0.23.2", "pyyaml", "colorednoise==1.1.1",
        "librosa==0.8.1", "catalyst", "torchaudio"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3.8",
    ],
    keywords=[],
)
