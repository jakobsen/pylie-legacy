import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pylie", # Replace with your own username
    version="0.0.1",
    author="Erik André Jakobsen",
    author_email="erik.a.jakobsen@gmail.com",
    description="A collection of Lie group integrators for autonomous ODEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakobsen/pylie",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.18']
)