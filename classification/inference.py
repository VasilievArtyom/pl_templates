#!/usr/bin/env python3
import argparse
import sys
import yaml
from pytorch_lightning import Trainer
from loguru import logger
from pytorch_lightning.strategies import DDPStrategy

from datasets.inference import create_inference_loader_dataset
from pl_models import TestModel


def load_config(config_path):
    with open(config_path, 'r') as input_file:
        config = yaml.safe_load(input_file)

    return config


def parse_args(args):
    parser = argparse.ArgumentParser(description='Template to inference networks with pytorch lightning.')

    parser.add_argument('config', help='path to yaml config file', default=None)
    return parser.parse_args(args)


@logger.catch
def inference(args=None):
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)
        config = load_config(args.config)
    else:
        config = args

    dataloader, dataset = create_inference_loader_dataset(
        config['annotations'],
        config['images'],
        config,
    )

    model = TestModel(
        config,
        dataloader,
        dataset
    )

    tester = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=False, **config['trainer']
    )
    tester.test(model)


if __name__ == "__main__":
    inference()
