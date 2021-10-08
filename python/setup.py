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
        "torch>=1.9.0",
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
