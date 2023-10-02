import copy
import json
import pickle
from typing import *

import torch
from torch.utils.data._utils.collate import default_collate
from openprompt.utils.logging import logger


class InputData(object):

    def __init__(self,
                 guid = None,
                 label=None,
                 # affordance=None,
                 explicit_sentence = "",
                 implicit_sentence = "",
                 box=[]
                ):
        self.guid = guid
        self.explicit_sentence = explicit_sentence
        self.implicit_sentence = implicit_sentence
        self.label = label
        # self.affordance = affordance
        box1 = []
        for i in box:
            box1.append(int(i))
        self.box = box1

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputData']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputData'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)