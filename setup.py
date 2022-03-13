from setuptools import setup

setup(
    name="deeptrees",
    version="1.0.0",
    description="Train networks of decision trees",
    long_description="Train networks of decision trees",
    packages=["deeptrees"],
    install_requires=["scikit-learn", "torch"],
    author="Alex Nichol",
    author_email="unixpickle@gmail.com",
    url="https://github.com/unixpickle/deeptrees",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
