# NEW_CALIFORNIA_EQUITY

This repository supports an empirical evaluation of the California Climate Investments (CCI) program, with a focus on programmatic efficiency, equity in funding distribution, and the institutional and spatial dimensions of implementation. Analyses center on greenhouse gas (GHG) reduction cost-effectiveness and the targeting of disadvantaged communities (DACs) across time, regions, and agencies.

## Repository Contents

- `cci_costperton_analysis.ipynb`: Hierarchical regression models assessing the relationship between inter-county collaboration, program scale, and cost per ton of GHG reduction, with agency and county fixed effects.
- `cci_time_series_analysis.ipynb`: Time-series models and descriptive trends in average funding, GHG reduction efficiency, and DAC share from 2015–2024.
- `cci_shareperdac_analysis.ipynb`: Regression analysis of predictors associated with higher shares of funding to disadvantaged communities.
- `cci_spatial_analysis.ipynb`: Exploratory spatial analysis of equity and efficiency metrics using geographic identifiers.
- `cci_spatial_analysis_winsorized_cleaned.ipynb`: Variant of the spatial analysis with outlier handling and cleaned inputs.
- `cci_programs_data_reduced.csv`: Cleaned dataset derived from the CCI Reporting & Tracking System (CCIRTS) used across all analyses.

## Analytical Goals

This project addresses three core questions:
1. How does inter-agency collaboration affect program efficiency?
2. How have efficiency and equity outcomes changed over time, particularly after 2020?
3. What spatial and institutional patterns explain variation in performance across CCI projects?

## Key Findings

- **Efficiency**: Projects involving more inter-county collaboration tend to reduce GHGs at lower cost, though gains taper in Southern California.
- **Temporal Shifts**: Average project funding has increased substantially since 2020, while efficiency has declined and equity outcomes have become more volatile.
- **Equity**: DAC targeting improved from 2015 to 2020 but declined sharply after 2022, despite increased funding.
- **Institutional and Geographic Effects**: Persistent variation in baseline efficiency and DAC allocation across agencies and counties suggests structural implementation constraints.

## Requirements

This repository uses Python 3.9+ and the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `geopandas` (for spatial analysis)
- `tabulate`

Install requirements via:

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NEW_CALIFORNIA_EQUITY.git
   cd NEW_CALIFORNIA_EQUITY
   ```

2. Open and run notebooks in this suggested sequence:
   - `cci_costperton_analysis.ipynb`
   - `cci_time_series_analysis.ipynb`
   - `cci_shareperdac_analysis.ipynb`
   - `cci_spatial_analysis.ipynb` or `cci_spatial_analysis_winsorized_cleaned.ipynb`

3. Visualizations and model summaries are embedded in the notebooks.

## Citation

> Adams, D. (2025). *Administrative Resilience Under Pressure: Environmental Governance and Institutional Adaptation*. Manuscript in progress.

## Contact

David P. Adams, Ph.D.  
California State University, Fullerton  
📧 dpadams@fullerton.edu  
🌐 https://dadams.io
📄 [Google Scholar](https://scholar.google.com/citations?user=Pg3KXfkAAAAJ&hl=en)