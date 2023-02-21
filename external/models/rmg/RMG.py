from tqdm import tqdm
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .RMGModel import RMGModel
from .sampler import *

import numpy as np
import pandas as pd


class RMG(RecMixin, BaseRecommenderModel):
    r"""
    Reviews Meet Graphs: Enhancing User and Item Representations for Recommendation with Hierarchical Attentive Graph Neural Network

    For further details, please refer to the `paper <https://aclanthology.org/D19-1494/>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 256, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_wcfm", "wcfm", "wcfm", 100, int, None),
            ("_wcfk", "wcfk", "wcfk", 3, int, None),
            ("_wa", "wa", "wa", 100, int, None),
            ("_scfm", "scfm", "scfm", 100, int, None),
            ("_scfk", "scfk", "scfk", 3, int, None),
            ("_sa", "sa", "sa", 100, int, None),
            ("_da", "da", "da", 100, int, None),
            ("_dau", "dau", "dau", 100, int, None),
            ("_factors", "factors", "f", 100, int, None),
            ("_uia", "uia", "uia", 100, int, None),
            ("_iua", "iua", "iua", 100, int, None),
            ("_ua", "ua", "ua", 100, int, None),
            ("_ia", "ia", "ia", 100, int, None),
            ("_dropout", "dropout", "d", 0.5, float, None),
            ("_loader", "loader", "l", 'WordsTextualAttributesPreprocessed', str, None)
        ]
        self.autoset_params()

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

        row, col = data.sp_i_train.nonzero()
        self.row = np.array(row)
        self.col = np.array(col)
        self.ratings = data.sp_i_train_ratings.data

        self._interactions_textual = self._data.side_information.WordsTextualAttributesPreprocessed
        self._interactions_textual.object.load_all_features()

        self._model = RMGModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            word_cnn_fea_maps=self._wcfm,
            word_cnn_fea_kernel=self._wcfk,
            word_att=self._wa,
            sent_cnn_fea_maps=self._scfm,
            sent_cnn_fea_kernel=self._scfk,
            sent_att=self._sa,
            doc_att=self._da,
            doc_att_u=self._dau,
            latent_size=self._factors,
            ui_att=self._uia,
            iu_att=self._iua,
            un_att=self._ua,
            in_att=self._ia,
            dropout_rate=self._dropout,
            max_reviews_user=self._interactions_textual.object.all_user_texts_shape[1],
            max_reviews_item=self._interactions_textual.object.all_item_texts_shape[1],
            max_sents=self._interactions_textual.object.all_user_texts_shape[2],
            max_sent_length=self._interactions_textual.object.all_user_texts_shape[3],
            max_neighbor=self._interactions_textual.object.user_to_item_shape[1],
            embed_vocabulary_features=self._interactions_textual.object.embed_vocabulary_features,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "RMG" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in generate_batch_data_random(self._interactions_textual.object.all_item_texts_features,
                                                        self._interactions_textual.object.all_user_texts_features,
                                                        self._interactions_textual.object.user_to_item_to_user_features,
                                                        self._interactions_textual.object.user_to_item_features,
                                                        self._interactions_textual.object.item_to_user_to_item_features,
                                                        self._interactions_textual.object.item_to_user_features,
                                                        self.col,
                                                        self.row,
                                                        self.ratings,
                                                        self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
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
                users = current_df['user'].tolist()
                inputs = [
                    self._interactions_textual.object.all_user_texts_features[users],
                    self._interactions_textual.object.user_to_item_to_user_features[users],
                    self._interactions_textual.object.user_to_item_features[users],
                    np.array(users)
                ]
                out_users = self._model.model_user(inputs, training=False)
                items = current_df['item'].tolist()
                inputs = [
                    self._interactions_textual.object.all_item_texts_features[items],
                    self._interactions_textual.object.item_to_user_to_item_features[items],
                    self._interactions_textual.object.item_to_user_features[items],
                    np.array(items)
                ]
                out_items = self._model.model_item(inputs, training=False)
                p = self._model.predict([out_users, out_items])
                predictions_val += p.numpy().tolist()
                t.update()
        test_len = len(self.df_test_rat)
        with tqdm(total=int(test_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, test_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, test_len)
                current_df = self.df_test_rat[offset:offset_stop]
                users = current_df['user'].tolist()
                inputs = [
                    self._interactions_textual.object.all_user_texts_features[users],
                    self._interactions_textual.object.user_to_item_to_user_features[users],
                    self._interactions_textual.object.user_to_item_features[users],
                    np.array(users)
                ]
                out_users = self._model.model_user(inputs, training=False)
                items = current_df['item'].tolist()
                inputs = [
                    self._interactions_textual.object.all_item_texts_features[items],
                    self._interactions_textual.object.item_to_user_to_item_features[items],
                    self._interactions_textual.object.item_to_user_features[items],
                    np.array(items)
                ]
                out_items = self._model.model_item(inputs, training=False)
                p = self._model.predict([out_users, out_items])
                predictions_test += p.numpy().tolist()
                t.update()
        return predictions_val, predictions_test

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
