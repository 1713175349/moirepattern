from setuptools import setup, find_packages

setup(
    name='moiregenerate',
    version='0.2.0',
    author="zhao luneng",
    author_email="1713175349@qq.com",
    description="Generate moiré patterns for 2D material bilayer structures",
    long_description="A Python package for generating moiré superlattice structures in 2D material bilayers. "
                    "Supports rotation angle scanning, supercell matching, and structure optimization.",
    license="MIT",
    keywords=["moire pattern", "2D materials", "bilayer", "twistronics", "supercell", "materials science"],
    packages=find_packages(),
    scripts=["bin/moiregenerate-cmd"],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "ase>=3.22.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "viz": [
            "jupyterlab>=3.0",
            "nglview>=3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "moiregenerate=moiregenerate.buildmoremodelcmd:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)