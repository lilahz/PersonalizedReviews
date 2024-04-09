from .ciao import CiaoDataset

DATASETS = {
    CiaoDataset.code(): CiaoDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
