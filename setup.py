import setuptools
import bida

def load_readme():
    with open("README.md", "r", encoding='utf8') as file:
        readme_text = file.read()
    return readme_text

def load_requirements():
    with open("requirements.txt", "r") as file:
        lines = [line.strip() for line in file]
    for i, line in enumerate(lines):
        if 'streamlit' in line:
            lines = lines[:i]
            break
    return lines

setuptools.setup(
    name="bida",
    version=bida.__version__,
    author="Pengfei Zhou",
    author_email="pfzhou@gmail.com",
    description="bida， 简单、易用、稳定、高效，便于扩展和集成的，大语言模型工程化开发框架",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pfzhou/bida",
    keywords=" ai aigc llm chat completion embedding",
    packages=setuptools.find_packages(include=["bida","bida.*"]),
    package_data={
        "":["*.json", "*.md"], 
    },
    include_package_data=True,
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    )
