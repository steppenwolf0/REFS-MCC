Library usage
=============

.. automodule:: refs_mcc
   :members:

Class details
-------------

.. autoclass:: refs_mcc.REFS_MCC
   :members:
   :undoc-members:
   :show-inheritance:

Usage example
-------------

.. code-block:: python

   from refs_mcc import REFS_MCC

   # construct with desired parameters
   refs_mcc = REFS_MCC(data="../data")

   # run the full pipeline
   refs_mcc.run()