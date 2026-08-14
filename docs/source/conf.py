import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src").resolve()))

pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"

with pyproject_path.open("rb") as f:
    data = tomllib.load(f)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'REFS-MCC'
copyright = '2026, Alejandro Lopez-Rincon, Alberto Tonda, Brigitta Varga, David Rojas-Velazquez'
author = 'Alejandro Lopez-Rincon, Alberto Tonda, Brigitta Varga, David Rojas-Velazquez'
release = data["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
