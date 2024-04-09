from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .ciao import CiaoDataset
from .ciao_beauty import CiaoBeautyDataset
from .ciao_games import CiaoGamesDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    CiaoDataset.code(): CiaoDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
