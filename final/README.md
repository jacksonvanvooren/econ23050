## Econ 23050 Final Project

This folder contains all the code written for my final project on the geographic distribution of (AI) patent citations.
Because of Github file limits, I provide links to the datasets used in `prepare_datasets.py` below. Make sure to change
the file path if running the code.

### Data Sources

The `ai_model_predictions` file can be found at [USPTO Artificial Intelligence Patent Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset).

All the following can be found and downloaded for free as .tsv files on [PatentsView](https://patentsview.org/download/data-download-tables), which is also provided by the USPTO.
- `g_assignee_disambiguated.tsv`
- `g_application.tsv`
- `g_location_disambiguated.tsv`
- `g_us_patent_citation.tsv`

### Overview

Each .py file corresponds to a section or sections of the paper.
- `prepare_datasets.py` involves all preprocessing and merges of different data sources.
- `intro_methods.py` includes code for graphs and charts in the intro and methodology sections. These show trends that have been empirically observed in the literature, and I briefly affirm these patterns with my datasets.
- `gini.py` calculates Gini coefficients across countries and time.
- `home_bias.py` develops the maps of spillover effects and includes distance analyses of patents and their citations.
- `top_decile.py` includes all regressions in the final results subsection of the paper.

### Miscellany

For any questions about the code or the project more generally, feel free to reach me at jacksonvanvooren@uchicago.edu.
