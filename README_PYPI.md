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
Citing REFS-MCC

If you use it in your research, please use the following BibTeX entry.

```bibtex
@article{ROJASVELAZQUEZ2025100757,
   title = {Matthews correlation coefficient-based feature ranking in recursive ensemble feature selection for high-dimensional and low-sample size data},
   journal = {Machine Learning with Applications},
   volume = {22},
   pages = {100757},
   year = {2025},
   issn = {2666-8270},
   doi = {https://doi.org/10.1016/j.mlwa.2025.100757},
   url = {https://www.sciencedirect.com/science/article/pii/S2666827025001409},
   author = {David Rojas-Velazquez and Aletta D. Kraneveld and Alberto Tonda and Alejandro Lopez-Rincon},
   keywords = {Feature selection, Machine learning, Biomarker discovery, Deep learning},
}
```

--------------------------------------------------------------------
Publications already using this method:

https://link.springer.com/article/10.1186/s12859-019-3050-8
https://www.nature.com/articles/s41598-023-50601-7
https://www.cambridge.org/core/journals/gut-microbiome/article/machine-learning-identifies-differences-between-breast-milk-and-formula-in-the-gut-microbiome/686680B29E1BF1FB2A7C2A093994E315
https://link.springer.com/article/10.1186/s12859-024-05639-3
https://www.sciencedirect.com/science/article/pii/S0378512224002809


--------------------------------------------------------------------