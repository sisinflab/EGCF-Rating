import typing as t
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class WordsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger

        self.user_reviews_path = getattr(ns, "user_reviews", None)
        self.item_reviews_path = getattr(ns, "item_reviews", None)
        self.user_item2id_path = getattr(ns, "user_item2id", None)
        self.item_user2id_path = getattr(ns, "item_user2id", None)
        self.user_doc_path = getattr(ns, "user_doc", None)
        self.item_doc_path = getattr(ns, "item_doc", None)
        self.w2v_path = getattr(ns, "w2v", None)

        self.user_reviews = None
        self.item_reviews = None
        self.user_item2id = None
        self.item_user2id = None
        self.user_doc = None
        self.item_doc = None
        self.w2v = None

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "WordsTextualAttributes"
        ns.object = self
        self.user_reviews = np.load(self.user_reviews_path)
        self.item_reviews = np.load(self.item_reviews_path)
        self.user_item2id = np.load(self.user_item2id_path)
        self.item_user2id = np.load(self.item_user2id_path)
        self.user_doc = np.load(self.user_doc_path)
        self.item_doc = np.load(self.item_doc_path)
        self.w2v = np.load(self.w2v_path)

        return ns
