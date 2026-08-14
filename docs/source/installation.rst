Installation
============

Install from PyPI (recommended)

Prerequisites
- Python 3.10+

1. (Optional) Create and activate a virtual environment

PowerShell (Windows):

.. code-block:: powershell

   python -m venv .venv
   & .\.venv\Scripts\Activate.ps1

Unix / macOS:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

2. Install the released package from PyPI:

.. code-block:: bash

   pip install refs-mcc

After installation a console script named ``refs-mcc`` is available. Use ``refs-mcc --help`` to see options.

Developer / editable install
----------------------------

If you want to work on the source and run the code from the repository, install in editable mode:

.. code-block:: bash

   pip install -e .

Build documentation

Install Sphinx and docs dependencies (recommended in the ``docs`` venv):

.. code-block:: bash

   pip install ".[docs]"

Build the docs

On Windows (from the ``docs`` folder):

.. code-block:: powershell

   make.bat html

On Unix/macOS:

.. code-block:: bash

   cd docs
   make html

Notes

- The package's console script is ``refs-mcc`` (entry point: ``refs_mcc:main``).
- When running code directly from the repository without installation, set ``PYTHONPATH=src`` or run files under ``src/`` (for example ``python src/refs_mcc.py ...``).
