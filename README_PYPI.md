# REFS-MCC
Recursive Ensemble Feauture Selection using Matthews Correlation Coefficient

--------------------------------------------------------------------
### Installation

```bash
   pip install refs-mcc
```

--------------------------------------------------------------------
### Usage

#### Command line

To run the full MCC-REFS pipeline just execute the installed console script:
```bash
   refs-mcc
```

To run it on a specific data folder, saving the results to an another specified location:
```bash
   refs-mcc --data data1 --output results1
```

For more information, run:
```bash
   refs-mcc --help
```

--------------------------------------------------------------------
#### Python library
It can also be used in a python code:

```python
    from refs_mcc import REFS_MCC
    REFS_MCC().run()
```

--------------------------------------------------------------------
### Input

If the location of the data folder is not specified (the `--data` argument), then by default, a folder named `data` needs to be next to the folder where the code is executed from. 

It needs to contain the following files:
- `data_0.csv`
- `features_0.csv`
- `labels.csv`

See an [example](https://github.com/steppenwolf0/REFS-MCC/tree/main/data) in the github repository. 

--------------------------------------------------------------------
### Output

If the location of the output folder is not specified (the `--output` argument), then the results will be saved into the same folder where the code is executed from.

The following folders and files will be created:
- run folders (`run0`, `run1`, `run2`, ..., `run{n-1}`, where n is the selected number of total runs, 10 by default)
- `best` folder containing:
   - `data_0.csv` - the transformed data
   - `features_0.csv` - the selected features
   - and more files with details
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
Examples of publications using this method (among others):

- [Lopez-Rincon, Alejandro, et al. "Automatic discovery of 100-miRNA signature for cancer classification using ensemble feature selection." BMC bioinformatics 20.1 (2019): 480.](https://link.springer.com/article/10.1186/s12859-019-3050-8)

- [Peralta-Marzal, Lucia N., et al. "A robust microbiome signature for autism spectrum disorder across different studies using machine learning." Scientific Reports 14.1 (2024): 814.](https://www.nature.com/articles/s41598-023-50601-7)

- [Liu, Ting Chia, et al. "Machine learning identifies differences between breast milk and formula in the gut microbiome." Gut Microbiome 7 (2026): e7.](https://www.cambridge.org/core/journals/gut-microbiome/article/machine-learning-identifies-differences-between-breast-milk-and-formula-in-the-gut-microbiome/686680B29E1BF1FB2A7C2A093994E315)

- [Rojas-Velazquez, David, et al. "Methodology for biomarker discovery with reproducibility in microbiome data using machine learning." BMC bioinformatics 25.1 (2024): 26.](https://link.springer.com/article/10.1186/s12859-024-05639-3)

- [Rojas-Velazquez, David, et al. "Understanding Parkinson's: The microbiome and machine learning approach." Maturitas 193 (2025): 108185.](https://www.sciencedirect.com/science/article/pii/S0378512224002809)

- [Varga, Brigitta, et al. "A systems microbiology framework for reproducible multi-dataset omics integration with application to long COVID." Frontiers in Systems Biology 6 (2026): 1873899.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13581729/)
--------------------------------------------------------------------