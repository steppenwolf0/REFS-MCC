# REFS-MCC
Recursive Ensemble Feauture Selection using Matthews Correlation Coefficient


--------------------------------------------------------------------
Instructions

- From the src folder run the refs-mcc (the default params are: 10 threads, 10 folds and 10 total runs):
```bash
   python refs-mcc.py
```

- For the full list of command line args:
```bash
   python refs-mcc.py --help
```

--------------------------------------------------------------------
Output

- run folders (`run0`, `run1`, ..., `run{n-1}`, where n is the selected number of total runs, 10 by default)
- `best` folder 
- `sumFig.pdf` & `sumFig.png`

--------------------------------------------------------------------