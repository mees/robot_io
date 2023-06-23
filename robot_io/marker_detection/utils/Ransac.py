import random

import numpy as np


class Ransac(object):
    """General purpose Ransac class.

    Given a function that estimates a model that fits the given data robustly.
    data_in: NxK, Contains N samples of input data (each sample is of dimension K).
    data_out: NxC, Contains N samples of output data (each sample is of dimension C).
    estimate_model_fct(data_in_ss, data_out_ss): Given data_in_ss in M'xK and data_out_ss in M'xC a model is estimated.
    score_model_fct(model, data_in_ss, data_out_ss): Calculates scores for each data point. Returns a vector of scores of length M'
    """

    def __init__(
        self,
        data_in,
        data_out,
        estimate_model_fct,
        score_model_fct,
        thresh,
        num_pts_needed,
        percentage_thresh=0.99,
        outlier_ratio=0.2,
    ):
        self.data_in = data_in
        self.data_out = data_out
        self.estimate = estimate_model_fct
        self.score = score_model_fct
        self.thresh = thresh
        self.num_pts_needed = num_pts_needed

        self.num_runs = np.ceil(np.log(1 - percentage_thresh) / np.log(1 - np.power(1 - outlier_ratio, num_pts_needed)))
        self.num_runs = int(self.num_runs)

    def run(self):
        inliers_best = None
        n, inliers = -float("inf"), None
        for i in range(self.num_runs):
            subset = random.sample(range(self.data_in.shape[0]), self.num_pts_needed)
            model = self.estimate(self.data_in[subset], self.data_out[subset])
            scores = self.score(model, self.data_in, self.data_out)
            inliers = scores < self.thresh
            num_inliers = np.sum(inliers)

            if n < num_inliers and num_inliers > 0:
                n = num_inliers
                inliers_best = inliers

        if inliers_best is None:
            return None

        # use best fit to calculate model from all inliers
        model_best = self.estimate(self.data_in[inliers_best], self.data_out[inliers_best])
        score_best = self.score(model_best, self.data_in, self.data_out)

        return n, score_best, model_best


if __name__ == "__main__":

    """Check number of iterations to run."""
    # for num_pts_needed in [2, 3, 8]:
    #     for outlier_ratio in [0.1, 0.5, 0.6]:
    #         tmp = Ransac([], [], [], 0.0,
    #                      num_pts_needed=num_pts_needed, outlier_ratio=outlier_ratio,
    #                      percentage_thresh=0.99)
    #         print(num_pts_needed, outlier_ratio, tmp.num_runs)
    # checks out with wikipedia

    """ Estimate lines with it. """
    x = np.arange(0, 10).astype(np.float32)
    y = 2 * x + 1
    y_m = np.copy(y)
    y_m[8] += 8.0  # outlier
    y_m[3] += 3.0  # outlier
    y_m += np.random.randn(*y.shape) * 0.5  # some noise

    def score(model, x, y):
        y_model = model[0] * x + model[1]
        return np.abs(y_model - y)

    def estimate(x, y):
        N = x.shape[0]  # num of samples
        A = np.ones((N, 2))
        B = np.zeros((N,))
        for i in range(N):
            A[i, 0] = x[i]
            A[i, 1] = 1.0
            B[i] = y[i]
        x = np.linalg.lstsq(A, B, rcond=None)[0].squeeze()
        return x

    est = Ransac(x, y_m, estimate, score, 0.1, 2)
    err, model = est.run()
    print("Finale model error", err)

    model_naive = estimate(x, y_m)
    y_model_naive = model_naive[0] * x + model_naive[1]

    y_model = model[0] * x + model[1]
    import matplotlib.pyplot as plt

    plt.plot(x, y, "g")
    plt.plot(x, y_m, "r")
    plt.plot(x, y_model_naive, "c:")
    plt.plot(x, y_model, "m:")
    plt.show()
