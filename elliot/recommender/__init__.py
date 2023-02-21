"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import BPRMF, BPRMF_batch, WRMF, PureSVD, MF, FunkSVD, PMF, LMF, NonNegMF, FM, LogisticMF, \
    FFM, BPRSlim, Slim, CML, FISM, SVDpp, MF2020, iALS, MF2020_batch
from .neural import DeepFM, DMF, NeuMF, NFM, GMF, NAIS, UserAutoRec, ItemAutoRec, ConvNeuMF, WideAndDeep, ConvMF, NPR

