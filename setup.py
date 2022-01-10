import setuptools

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="StarkShift",
    version="1.0.1",
    author="Christoph Fischer",
    author_email="fischchr@phys.ethz.ch",
    description="A module for calculating ac Stark shifts on alkali and alkaline-earth atoms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git+https://gitlab.ethz.ch/tiqi-projects/optical-trap/StarkShift",
    packages=['StarkShift'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.6',
        'sympy>=1.7',
        'pint>=0.16',
        'pyshtools>=4.7',
        'atomphys>=0.0.4',
        'ARC-Alkali-Rydberg-Calculator>=3.0'
    ],
    zip_safe=False
)
