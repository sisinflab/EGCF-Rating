import typing as t
from scipy import sparse
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SentimentInteractionsTextualAttributesUUII(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.ii_dot_path = getattr(ns, "ii_dot", None)
        self.ii_max_path = getattr(ns, "ii_max", None)
        self.ii_min_path = getattr(ns, "ii_min", None)
        self.ii_avg_path = getattr(ns, "ii_avg", None)
        self.uu_dot_path = getattr(ns, "uu_dot", None)
        self.uu_max_path = getattr(ns, "uu_max", None)
        self.uu_min_path = getattr(ns, "uu_min", None)
        self.uu_avg_path = getattr(ns, "uu_avg", None)

        self.ii_rat_dot_path = getattr(ns, "ii_rat_dot", None)
        self.ii_rat_max_path = getattr(ns, "ii_rat_max", None)
        self.ii_rat_min_path = getattr(ns, "ii_rat_min", None)
        self.ii_rat_avg_path = getattr(ns, "ii_rat_avg", None)
        self.uu_rat_dot_path = getattr(ns, "uu_rat_dot", None)
        self.uu_rat_max_path = getattr(ns, "uu_rat_max", None)
        self.uu_rat_min_path = getattr(ns, "uu_rat_min", None)
        self.uu_rat_avg_path = getattr(ns, "uu_rat_avg", None)

        self.uu_no_coeff_path = getattr(ns, "uu_no_coeff", None)
        self.uu_rat_no_coeff_path = getattr(ns, "uu_rat_no_coeff", None)
        self.ii_no_coeff_path = getattr(ns, "ii_no_coeff", None)
        self.ii_rat_no_coeff_path = getattr(ns, "ii_rat_no_coeff", None)

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "SentimentInteractionsTextualAttributesUUII"
        ns.object = self

        return ns

    def get_features(self):
        if self.ii_dot_path:
            yield 'ii_dot', sparse.load_npz(self.ii_dot_path)
        if self.ii_max_path:
            yield 'ii_max', sparse.load_npz(self.ii_max_path)
        if self.ii_min_path:
            yield 'ii_min', sparse.load_npz(self.ii_min_path)
        if self.ii_avg_path:
            yield 'ii_avg', sparse.load_npz(self.ii_avg_path)

        if self.uu_dot_path:
            yield 'uu_dot', sparse.load_npz(self.uu_dot_path)
        if self.uu_max_path:
            yield 'uu_max', sparse.load_npz(self.uu_max_path)
        if self.uu_min_path:
            yield 'uu_min', sparse.load_npz(self.uu_min_path)
        if self.uu_avg_path:
            yield 'uu_avg', sparse.load_npz(self.uu_avg_path)

        if self.ii_rat_dot_path:
            yield 'ii_rat_dot', sparse.load_npz(self.ii_rat_dot_path)
        if self.ii_rat_max_path:
            yield 'ii_rat_max', sparse.load_npz(self.ii_rat_max_path)
        if self.ii_rat_min_path:
            yield 'ii_rat_min', sparse.load_npz(self.ii_rat_min_path)
        if self.ii_rat_avg_path:
            yield 'ii_rat_avg', sparse.load_npz(self.ii_rat_avg_path)

        if self.uu_rat_dot_path:
            yield 'uu_rat_dot', sparse.load_npz(self.uu_rat_dot_path)
        if self.uu_rat_max_path:
            yield 'uu_rat_max', sparse.load_npz(self.uu_rat_max_path)
        if self.uu_rat_min_path:
            yield 'uu_rat_min', sparse.load_npz(self.uu_rat_min_path)
        if self.uu_rat_avg_path:
            yield 'uu_rat_avg', sparse.load_npz(self.uu_rat_avg_path)

        if self.ii_no_coeff_path:
            yield 'ii_no_coeff', sparse.load_npz(self.ii_no_coeff_path)
        if self.ii_rat_no_coeff_path:
            yield 'ii_rat_no_coeff', sparse.load_npz(self.ii_rat_no_coeff_path)

        if self.uu_no_coeff_path:
            yield 'uu_no_coeff', sparse.load_npz(self.uu_no_coeff_path)
        if self.uu_rat_no_coeff_path:
            yield 'uu_rat_no_coeff', sparse.load_npz(self.uu_rat_no_coeff_path)
