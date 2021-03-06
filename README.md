# Preamble

This code is the one used to generate results presented in the paper
*Cost-Aware Early Classification of Time Series*.
When using this code, please cite:
```
@InProceedings{costaware2016,  
    authors={Romain Tavenard and Simon Malinowski},
    title={Cost-Aware Early Classification of Time Series},
    booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery},
    pages = {632-647},
    year={2016}
}
```

# Requirements
For this code to run properly, the following python packages should be installed:
```
numpy  
scipy  
sklearn
```

Also, if one wants to run experiments on the UCR dataset, she should download it from
[here](http://www.cs.ucr.edu/~eamonn/time_series_data/) and paste it (preserving its subfolder structure) to `datasets/ucr`.

# Running
## Baseline (Dachraoui et al., ECML 2015)
To run the baseline on dataset `FISH` with $\beta=0.001$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/classification/
EXECUTABLE=${SOURCEDIR}/classification/baseline_ucr.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 0.001
```

## 2Step
To run the _2Step_ method on dataset `FISH` with $\beta=0.001$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/classification/
EXECUTABLE=${SOURCEDIR}/classification/2step_classif_ucr.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 0.001
```

## NoCluster
To run the _NoCluster_ method on dataset `FISH` with $\beta=0.001$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/classification/
EXECUTABLE=${SOURCEDIR}/classification/nocluster_ucr.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 0.001
```
