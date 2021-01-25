import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="batzzang",
    version="0.0.1",
    author="Inhyuk Na",
    author_email="ina@dblab.postech.ac.kr",
    description="pytorch lazy batching library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inyukwo1/batzzang",
    packages=setuptools.find_packages(),
    install_requires=[
          'torch'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)