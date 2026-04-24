import os

def directory_check(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'predictions')):
        os.makedirs(os.path.join(data_dir, 'predictions'))

    if not os.path.exists(os.path.join(data_dir, 'results')):
        os.makedirs(os.path.join(data_dir, 'results'))

    if not os.path.exists(os.path.join(data_dir, 'ML_models')):
        os.makedirs(os.path.join(data_dir, 'ML_models'))

    if not os.path.exists(os.path.join(data_dir, 'DL_models')):
        os.makedirs(os.path.join(data_dir, 'DL_models'))
