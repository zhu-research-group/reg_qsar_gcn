# Evaluating Drug-Drug Interactions by Constructing Advanced Machine Learning Models of ABC Transporter Inhibitors



## Setup

The QSAR and GCN workflows access files by referencing directories stored within environment variables saved in the computer's operating system. To run these models, save the directory containing the modeling datasets and scripts in an environment variable.

Running this code requires an Anaconda distribution for Python. Navigate to the directory containing the provided files and run the following command line prompt to create the necessary Python environment.

```conda env create -f environment.yml```

Activate the new environment once the required installations are complete.

```conda activate reg_qsar_gcn```

Setup has been previously described in further detail below:

_**Automatic Quantitative Structure-Activity Relationship Modeling to Fill Data Gaps in High-Throughput Screening.**_ Methods Mol Biol, 2022. (https://doi.org/10.1007/978-1-0716-2213-1_16)

## Training the models
Once in the active environment, the QSAR and GCN models can be run in the command line. Each script and function accept several parameters.

#### build_ml_regressors.py:
1. `-ds`: Name of your .sdf file
2. `-dd`: Name of the environmental variable of project directory
3. `-nc`: Name of the identifier column in .sdf file
4. `-ep`: Name of the endpoint column in .sdf file
5. `-ts`: Size of the test set split
6. `-ns`: Number of splits for cross-validation
7. `-f`: Features with which to build the models

**Example Use:** ```python build_ml_regressors.py -ds pgp_pic50_modeling_set_20250807_SC_CSP -dd {YOUR_ENVIRONMENT_VARIABLE_NAME} -nc Compound_ID -ep ACTIVITY -ts 0.15 -ns 5 -f MACCS```

#### make_predictions.py
1. `-ds`: Name of your .sdf file
2. `-dd`: Name of the environmental variable of project directory
3. `-nc`: Name of the identifier column in .sdf file
4. `-ep`: Name of the endpoint column in .sdf file
5. `-ps`: Name of your prediction set .sdf file
6. `-a`: Algorithms to use for model predictions
7. `-f`: Features to use for model predictions

**Example Use:** ```python make_predictions.py -ds pgp_pic50_modeling_set_20250807_SC_CSP -dd {YOUR_ENVIRONMENT_VARIABLE_NAME} -nc Compound_ID -ep ACTIVITY -ps {PREDICTION_SET} -a "rfr,svr,xgb,knnr" -f ECFP6```

#### build_gcn_regressors.py:
1. `-ds`: Name of your .sdf file
2. `-dd`: Name of the environmental variable of project directory
3. `-nc`: Name of the identifier column in .sdf file
4. `-ep`: Name of the endpoint column in .sdf file
5. `-ns`: Number of splits for cross-validation
6. `-ts`: Size of the test set split
7. `-cl`: Size and shape of graph convolutional layers
8. `-do`: Dropout for each layer
9. `-bs`: Batch size used by model
10. `-ne`: Number of times to iterate over the full dataset during model training
11. `-rm`: Include in command line to reuse existing model. If argument is missing, a new model will overwrite the existing model

**Example Use:** ```python build_gcn_regressors.py -ds pgp_pic50_modeling_set_20250807_SC_CSP -dd {YOUR_ENVIRONMENT_VARIABLE_NAME} -nc Compound_ID -ep ACTIVITY -ns 5 -ts 0.15 -cl 64 128 64 -do 0.0 -bs 16 -ne 50 -rm```

#### make_gcn_predictions.py:
1. `-ds`: Name of your .sdf file
2. `-dd`: Name of the environmental variable of project directory
3. `-nc`: Name of the identifier column in .sdf file
4. `-ps`: Name of your prediction set .sdf file

**Example Use:** ```python make_gcn_predictions.py -ds pgp_pic50_modeling_set_20250807_SC_CSP -dd {YOUR_ENVIRONMENT_VARIABLE_NAME} -nc Compound_ID -ps {PREDICTION_SET}```
