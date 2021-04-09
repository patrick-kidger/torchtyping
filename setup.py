import os
import re
import setuptools
import sys

here = os.path.realpath(os.path.dirname(__file__))


name = "torchtyping"

# for simplicity we actually store the version in the __version__ attribute in the
# source
with open(os.path.join(here, name, "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

author = "Patrick Kidger"

author_email = "contact@kidger.site"

description = "Runtime type annotations for the shape, dtype etc. of PyTorch Tensors. "

with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()

url = "https://github.com/patrick-kidger/torchtyping"

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Framework :: Pytest",
]

user_python_version = sys.version_info

python_requires = ">=3.7.0"

install_requires = ["torch>=1.7.0", "typeguard>=2.11.1"]

if user_python_version < (3, 9):
    install_requires += ["typing_extensions==3.7.4.3"]

entry_points = dict(pytest11=["torchtyping = torchtyping.pytest_plugin"])

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    entry_points=entry_points,
    packages=[name],
)
