CLI Usage
=========

REFS-MCC provides a single packaged console script entry point with a simple command-line interface.

Examples

Run the full REFS pipeline (installed console script ``refs-mcc``):

.. code-block:: powershell

   # From repository root (Windows PowerShell) after installing the package
   refs-mcc --data data

On different data folders, saving the result to different locations:

.. code-block:: powershell

   refs-mcc --data data1 --output results1
   refs-mcc --data data2 --output results2
   refs-mcc --data data3 --output results3

Check the help for the rest of the options:

.. code-block:: powershell

   refs-mcc --help

Output layout

- Scripts write their results into the provided ``--output`` folder.
- A typical run creates a ``best/`` subfolder containing ``data_0.csv``, ``features_0.csv``, ``labels.csv``, ``sum.csv``, and other artifacts.

If you installed the package (`pip install -e .`) and want to run the modules as packages, ensure the package entry points are configured or run via `python -m` with `PYTHONPATH=src`.
