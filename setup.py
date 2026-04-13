"""Setup configuration for WSN DDQN Training Platform."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wsn-ddqn",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Q-Learning for WSN Scheduling Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wsn-ddqn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.3.0",
            "flake8==6.0.0",
            "mypy==1.3.0",
            "pytest==7.4.0",
            "pytest-cov==4.1.0",
        ],
        "docs": [
            "sphinx==5.3.0",
            "sphinx-rtd-theme==1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wsn-train=scripts.train:main",
        ],
    },
)
