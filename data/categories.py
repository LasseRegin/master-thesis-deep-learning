
from __future__ import print_function

import os
import json
import pickle
import hashlib
import collections

import editdistance

dir = os.path.dirname(os.path.abspath(__file__))


class CategoryDeterminer:
    """
        Defines categories based on given tags.

        NOTE: This has been modified due to ownership issues, hence if this class
              is used for new data, the `category_dict.json` must be modified
              to contain the categories for the new tags.
    """
    category_dict_filepath = os.path.join(dir, 'umls/category_dict.json')

    def __init__(self, cui2info_filepath='umls/cui2info.pkl', verbose=False):
        self.verbose = verbose

        with open(self.category_dict_filepath, 'r') as f:
            self.category_dict = json.load(f)

    def clean_alias(self, string):
        return string.lower().strip()

    def define_categories(self, categories):
        new_categories = set()
        for category in categories:
            new_categories.add(self.lookup_category(category))

        return list(sorted(new_categories))

    def lookup_category(self, category):
        category = self.clean_alias(category)
        return self.category_dict[category]
