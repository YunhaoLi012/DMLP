from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent

VERSION = '1.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

setup(
       # 名称必须匹配文件名 'verysimplemodule'
        name="DMLP", 
        version=VERSION,
        author="YunhaoLi, Jieqi Liu",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=[], 
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)