# Adapted from the original Cohen's smoothing paper:
# Certified Adversarial Robustness via Randomized Smoothing
# Accompanying source code: https://github.com/locuslab/smoothing/blob/master/code/core.py

from typing import Tuple

import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch


class ClassifierSmoothing(object):
    """A smoothed classifier g """

    # to abstain, ClassifierSmoothing returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x 1] (logistic regression)
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, z: torch.tensor, n0: int, n: int, alpha: float, sampling_batch_size: int) -> Tuple[int, float]:
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        be robust within a L2 ball of radius R around x.
        :param z: the input [batch size = 1 x latent dimension]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param sampling_batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        assert z.size(0) == 1
        self.base_classifier.eval()
        # draw samples of f(x + epsilon)
        counts_selection = self._counts(z, n0, sampling_batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._counts(z, n, sampling_batch_size)

        assert counts_selection.ndim == 1 and counts_selection.shape[0] == self.num_classes
        assert counts_estimation.ndim == 1 and counts_estimation.shape[0] == self.num_classes

        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return ClassifierSmoothing.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, z: torch.tensor, n: int, alpha: float, sampling_batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param z: the input [batch_size = 1 x latent dimension]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param sampling_batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        assert z.size(0) == 1
        self.base_classifier.eval()
        counts_estimation = self._counts(z, n, sampling_batch_size)
        assert counts_estimation.ndim == 1 and counts_estimation.shape[0] == self.num_classes
        top2 = counts_estimation.argsort()[::-1][:2]
        count1 = counts_estimation[top2[0]]
        count2 = counts_estimation[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return ClassifierSmoothing.ABSTAIN
        else:
            return top2[0]

    def _counts(self, z: torch.tensor, num: int, sampling_batch_size: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param z: the input [batch_size = 1 x latent dimension]
        :param num: number of samples to collect
        :param sampling_batch_size:
        :return: a torch.tensor of shape [batch_size x num_classes] containing the per-class counts for each sample
                 from the batch
        """
        assert z.size(0) == 1
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            z_repeated = z.repeat_interleave(min(sampling_batch_size, num), dim=0)
            while num > 0:
                cur_num_copies = min(sampling_batch_size, num)
                num -= cur_num_copies

                if cur_num_copies != z_repeated.size(0):
                    z_repeated = z.repeat_interleave(cur_num_copies, dim=0)

                assert z_repeated.size(0) == z.size(0) * cur_num_copies and z_repeated.size(1) == z.size(1)
                noise = torch.randn_like(z_repeated, device=z_repeated.device) * self.sigma

                logits = self.base_classifier(z_repeated + noise)
                predictions = logits.argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    @staticmethod
    def _count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    @staticmethod
    def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
