# Pattern Recognition - Assignment 2 - Group 10

## Requirements
All code is written in Python. All required packages and libraries can be installed through the requirements file:

```bash
pip install -r requirements.txt
```

## Running the code
The entire pipeline can be run through the *main.ipynb* Python Notebook file. This file is divided into sections corresponding to the components of the pipeline. In each section, the code for the performed experiments can be found. Code that was used early on in the process (like parameter sweeps) may have been commented out. The data, algorithms, and functions for each component can be found in their corresponding subfolder. *Note that the data needs to be in the folders mentioned below, otherwise the pipeline does not work.*

### Data functions
Functions regarding data loading etc. can be found in *raw_data/data_functions.py*

### Genes data
- Data is expected to be in *raw_data/genes/data.csv*
- Labels are expected to be in *raw_data/genes/labels.csv*

### Image data
- Animal images are expected to be in *raw_data/BigCats/[Animal]/*

### Feature selection
- Feature selection algorithms can be found in *feature_selection/[algorithm].py*

### Classification
- Classification algorithms can be found in *classification/[algorithm].py*

### Clustering
- Clustering algorithms can be found in *clustering/[algorithm].py*

### Visualization
- Functions that were used for data visualization can be found in *visualization/visualization.py*