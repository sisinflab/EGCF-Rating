"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm
import random

from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .SVDppModel import SVDppModel
from elliot.recommender.recommender_utils_mixin import RecMixin
import pandas as pd


class SVDpp(RecMixin, BaseRecommenderModel):
    r"""
    SVD++

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/1401890.1401944>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg_w: Regularization coefficient for latent factors
        reg_b: Regularization coefficient for bias

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SVDpp:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 50
          lr: 0.001
          reg_w: 0.1
          reg_b: 0.001
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, None, None),
            ("_batch_eval", "batch_eval", "batch_eval", 512, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_lambda_weights", "reg_w", "reg_w", 0.1, None, None),
            ("_lambda_bias", "reg_b", "reg_b", 0.001, None, None),
        ]
        self.autoset_params()

        np.random.seed(self._seed)
        random.seed(self._seed)

        self.df_val_rat = pd.DataFrame(columns=['user', 'item', 'rating'])
        self.df_test_rat = pd.DataFrame(columns=['user', 'item', 'rating'])

        # create dataframes for validation and test set with user-item-rating
        idx = 0
        for k, v in data.val_dict.items():
            for kk, vv in v.items():
                self.df_val_rat = pd.concat([self.df_val_rat,
                                             pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        idx = 0
        for k, v in data.test_dict.items():
            for kk, vv in v.items():
                self.df_test_rat = pd.concat([self.df_test_rat,
                                              pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        self.df_val_rat = self.df_val_rat.astype({'user': int, 'item': int, 'rating': float})
        self.df_test_rat = self.df_test_rat.astype({'user': int, 'item': int, 'rating': float})

        self.df_val_rat['user'] = self.df_val_rat['user'].map(data.public_users)
        self.df_val_rat['item'] = self.df_val_rat['item'].map(data.public_items)
        self.df_test_rat['user'] = self.df_test_rat['user'].map(data.public_users)
        self.df_test_rat['item'] = self.df_test_rat['item'].map(data.public_items)

        # remove interactions whose user and/or item is not in the training set
        self.df_val_rat = self.df_val_rat[self.df_val_rat['user'] <= self._num_users - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['user'] <= self._num_users - 1]
        self.df_val_rat = self.df_val_rat[self.df_val_rat['item'] <= self._num_items - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['item'] <= self._num_items - 1]

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._sampler = Sampler(self._batch_size, self._data.transactions, self._data.sp_i_train_ratings)

        self._model = SVDppModel(self._num_users, self._num_items, self._factors,
                                 self._lambda_weights, self._lambda_bias, self._learning_rate, self._seed)

    @property
    def name(self):
        return "SVDpp" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            np.random.shuffle(edge_index)
            edge_index = edge_index.astype(int)

            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_test = []
        predictions_val = []
        val_len = len(self.df_val_rat)
        with tqdm(total=int(val_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, val_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, val_len)
                current_df = self.df_val_rat[offset:offset_stop]
                current_ratings = np.array(self._data.sp_i_train_ratings.todense()[current_df['user'].tolist()])
                p = self._model.predict(
                    inputs=(np.array(current_df['user'].tolist()),
                            np.array(current_df['item'].tolist()), current_ratings))
                predictions_val += p.numpy().tolist()
                t.update()
        test_len = len(self.df_test_rat)
        with tqdm(total=int(test_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, test_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, test_len)
                current_df = self.df_test_rat[offset:offset_stop]
                current_ratings = np.array(self._data.sp_i_train_ratings.todense()[current_df['user'].tolist()])
                p = self._model.predict(
                    inputs=(np.array(current_df['user'].tolist()),
                            np.array(current_df['item'].tolist()), current_ratings))
                predictions_test += p.numpy().tolist()
                t.update()
        return predictions_val, predictions_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0.0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            predictions_val, predictions_test = self.get_recommendations()
            true_val, true_test = self.df_val_rat['rating'].to_numpy(), self.df_test_rat['rating'].to_numpy()
            result_dict = self.evaluator.eval_error(np.array(predictions_val), true_val, np.array(predictions_test),
                                                    true_test)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.5f}')
            else:
                self.logger.info(f'Finished')

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][0]["val_results"][self._validation_metric]

    def get_loss(self):
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return min([r[0]["val_results"][self._validation_metric] for r in self._results])

    def get_best_arg(self):
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmin([r[0]["val_results"][self._validation_metric] for r in self._results])
        return val_results
