# REFS-MCC
Recursive Ensemble Feauture Selection using Matthews Correlation Coefficient

--------------------------------------------------------------------
Installation

```bash
   pip install refs-mcc
```

After installation it can be used as the following (using the default parameters):
```python
    from refs_mcc import REFS_MCC
    REFS_MCC().run()
```

Or from CLI. For more information, run:
```bash
   refs-mcc --help
```
 
--------------------------------------------------------------------
Input

Next to the folder where the code is executed from, a `data` folder needs to be present with the following files:
- `data_0.csv`
- `features_0.csv`
- `ids.csv`
- `labels.csv`

--------------------------------------------------------------------
Output

The following folders and files will be created:
- run folders (`run0`, `run1`, ..., `run{n-1}`, where n is the selected number of total runs, 10 by default)
- `best` folder 
- `sumFig.pdf` & `sumFig.png`

--------------------------------------------------------------------