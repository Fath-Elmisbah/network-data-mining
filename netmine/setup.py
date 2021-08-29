import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dot.energy.netmine",  # Replace with your own username
    version="0.0.1",
    author="Fathelrahman Elmisbah",
    author_email="Fath.Elmisbah@gmail.com",
    description="A package that provide useful tools to mine network data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fath-Elmisbah/network-data-mining/tree/main/netmine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
