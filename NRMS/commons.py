import os

from utils import verify_folder


class Commons:
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    SHARED_PATH = '/sise/bshapira-group/lilachzi/models/nrms'
    CSVS_PATH = '/sise/bshapira-group/lilachzi/csvs'
    ANALYSIS_PATH = os.path.join(BASE_PATH, 'analysis')
    MODELS_PATH = os.path.join(SHARED_PATH, 'models')
    SAVED_MODELS_PATH = os.path.join(SHARED_PATH, 'saved_models')
    PREDICTIONS_PATH = os.path.join(MODELS_PATH, 'predictions')
    LOGS_PATH = os.path.join(BASE_PATH, 'logs')

    verify_folder(MODELS_PATH)
    verify_folder(PREDICTIONS_PATH)
    verify_folder(LOGS_PATH)

    CATEGORIES = ['Baby', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics',
                  'Musical_Instruments', 'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games']
