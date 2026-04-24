import pandas as pd
from numpy import inf
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


def generate_molecules(file, endpoint=None):
    """
    Takes in the path to a QSAR-ready .sdf file and generates a list of rdkit molecule objects
    Returns training and external validation sets without molecules that could not be created from SMILES strings or
    molecules with no information about the desired endpoint
    Returns prediction sets without molecules that could not be created from SMILES strings

    :param dataset_name: String representing dataset name in .sdf file
    :param data_dir: The project directory containing the dataset
    :param endpoint: Desired binary or continuous endpoint with threshold to be modeled (for training sets and external
    validation sets, defaults to None); Enter None for prediction sets

    :return molecules: List of rdkit molecule objects
    """

    if endpoint is None:
        # Returns rdkit Mol objects for molecules in prediction set .sdf files that were able to be generated
        molecules = [mol for mol in SDMolSupplier(file) if mol is not None]

        # Checks for appropriate output
        assert molecules != [], f'No molecules could be generated from the dataset provided ({file}).'

        return molecules

    else:
        # Returns rdkit Mol objects for molecules in training and evaluation set .sdf files that were able to be
        # generated and have information pertaining to the desired endpoint
        molecules = [mol for mol in SDMolSupplier(file) if mol is not None]
                     # and mol.HasProp(endpoint) and mol.GetProp(endpoint) not in ['NA', 'nan', '']]

        # Checks for appropriate output
        assert molecules != [], f'No molecules could be generated with the given endpoint ({endpoint})'

        return molecules


def calc_rdkit(molecules, name_col='CID'):
    """
    Calculates rdkit descriptors of all molecules in a list

    :param molecules: List of rdkit Mol objects (type: list)
    :param name_col: Property of all objects in molecules to use as index of descriptor matrix, such as the
    type of ID assigned to the compounds eg. CID, CASRN, REGISTRY NUMBER (default: 'CID') (type: str)
    :return desc_matrix: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    assert None not in molecules
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

    X = pd.DataFrame([calculator.CalcDescriptors(mol) for mol in molecules],
                     index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules],
                     columns=list(calculator.GetDescriptorNames()))

    desc_matrix = X.fillna(X.mean())
    desc_matrix = desc_matrix.loc[:, ~desc_matrix.isin([inf, -inf]).any(axis=0)]
    assert len(desc_matrix.columns) != 0

    return desc_matrix


def calc_maccs(molecules, name_col='CID'):
    """
    Takes in a list of rdkit molecules and returns MACCS Keys fingerprints for a list of rdkit molecules

    :param molecules: List of rdkit molecules with no None values (default: 'CID') (type: list)
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property
    of all molecules (type: str)
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    assert None not in molecules
    data = []

    for mol in molecules:
        maccs = [float(x) for x in MACCSkeys.GenMACCSKeys(mol)]
        data.append(maccs)

    return pd.DataFrame(data, [mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_ecfp6(molecules, name_col='CID'):
    """
    Takes in a list of rdkit molecules and returns ECFP6 fingerprints for a list of rdkit molecules

    :param molecules: List of rdkit molecules with no None values (default: 'CID') (type: list)
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property
    of all molecules (type: str)
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    assert None not in molecules
    data = []

    for mol in molecules:
        ecfp6 = [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)]
        data.append(ecfp6)

    return pd.DataFrame(data, [mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_fcfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns FCFP6 fingerprints for a list of rdkit molecules

    :param molecules: List of rdkit molecules with no None values (default: 'CID') (type: list)
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property
    of all molecules (type: str)
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'

    data = []

    for mol in molecules:
        fcfp6 = [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)]
        data.append(fcfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])
