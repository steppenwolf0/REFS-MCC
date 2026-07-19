# REFS-MCC
Recursive Ensemble Feauture Selection using Matthews Correlation Coefficient

--------------------------------------------------------------------
Installation

```bash
   pip install refs-mcc
```

After installation it can be used from CLI, for more information, run:
```bash
   refs-mcc --help
```

Or from python code:
```python
    from refs_mcc import REFC_MCC

    if __name__ == '__main__': 
        REFC_MCC().run()
```
 
--------------------------------------------------------------------
Input

Next to the folder where the code is executed from, a `data` folder needs to be present with the following:
- `data_0.csv`
- `features_0.csv`
- `ids.csv`
- `labels.csv`

--------------------------------------------------------------------
Output

- run folders (`run0`, `run1`, ..., `run{n-1}`, where n is the selected number of total runs, 10 by default)
- `best` folder 
- `sumFig.pdf` & `sumFig.png`

--------------------------------------------------------------------