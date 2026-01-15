from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-momentum-trading",
    version="0.1.0",
    author="Rajeshwar Vempaty",
    author_email="vrsanurag@gmail.com",
    description="End-to-end ML momentum trading system with Zerodha Kite Connect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajeshwar-vempaty/ml-momentum-trading",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "kiteconnect>=4.2.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
