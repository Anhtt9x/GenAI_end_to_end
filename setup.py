from setuptools import find_packages, setup

setup(
    name="generator",
    version="0.0.0",
    description="A GenAi demo",
    author="Anhtt9x",
    author_email="anhtt454598@gmail.com",
    install_requires=["openai","langchain", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages(where="src"),
    package_dir={"":"src"}
)