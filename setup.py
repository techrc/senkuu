import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="senkuu",
    version="0.1.0",
    author="iqianshuai",
    author_email="iqianshuai@163.com",
    description="Deep Learning for beginner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iqianshuai/senkuu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
