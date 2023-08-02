import ot
import numpy as np
from scipy.special import softmax
from wasscal.metrics import calibration_error

def sinkhornTransport(source, target, bins):
    """
    :param source:
    :param target:
    :param bins:
    :return: OT Plan as matrix

    G0 is a.shape[0] x b.shape[0] matrix where G0[i,j] is mass in a[i] to b[j]
    """

    # Equivalent to
    M = ot.dist(bins.reshape(-1, 1), bins.reshape(-1, 1))
    M /= M.max()
    G0 = ot.emd(source, target, M**2)
    # G0 = ot.emd_1d(bins, bins, source, target)
    return G0



def estimate_classwise_density(discrete_belief, grid):
    """
    Compute the class density on scores that have been discretized to fit into grid
    :param discrete_belief:
    :param grid:
    :return:
    """
    bin_tally = np.digitize(discrete_belief, grid, right=True)
    density = np.bincount(bin_tally, minlength=grid.size)/bin_tally.size
    return density

def approximate_calibrated_density(classwise_density, grid, y, k, pos_case=True):

    if pos_case:
        likelihood = 1 / np.mean(y == k)
    else:
        likelihood = 1 / np.mean(y != k)

    prior = np.multiply(classwise_density, grid)
    dist = prior * likelihood
    return dist / dist.sum()


def discretize_scores(scores, eta=2):
    """
    Discretize scores to the level of prevision described by the precision param
    :param scores:
    :param eta:
    :return:
    """
    eta = int(10 ** eta) + 1
    grid = np.linspace(0, 1, eta)
    discrete = grid[np.searchsorted(grid, scores)]

    return discrete, grid


def apply_transport_plan(scores, grid, plan):
    """
    the division line lifted from https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    to solve divide by zero errors
    """
    marginal = plan.sum(axis=1)[:,np.newaxis]
    conditional_plan = np.divide(plan, marginal, out=np.zeros_like(plan), where=marginal != 0)
    empty = np.empty(shape=scores.shape)
    for i, ri in enumerate(conditional_plan):
        indexes = np.where(scores == grid[i])[0]
        expect_count = np.round(ri * indexes.size)

        #Little heuristic to get the rounded histograms as close to one another as possible
        while expect_count.sum() > indexes.size:
            max_vio = np.argmax(expect_count/indexes.size - ri)
            expect_count[max_vio] -= 1

        while expect_count.sum() < indexes.size:
            max_vio = np.argmin(expect_count / indexes.size - ri)
            expect_count[max_vio] += 1

        start_marker = 0
        cumulative = np.cumsum(expect_count)
        for i in expect_count.nonzero()[0]:
            stop_marker = int(start_marker + cumulative[i])
            change_ix = indexes[start_marker:stop_marker]
            empty[change_ix] = grid[i]
            start_marker = stop_marker

    return empty


def compute_kantorovich_plan(a, b):
    assert a.size == b.size, "oops"

    # bin positions
    x = np.arange(a.size, dtype=np.float64)

    # loss matrix
    M = ot.dist(x.reshape(-1, 1), x.reshape(-1, 1))
    M /= M.max()

    return ot.emd(a, b, M)

def transport_plan_k_calibrated(pred, y, k, eta=2):
    discrete_scores, grid = discretize_scores(pred[:, k], eta=eta)
    belief_density = estimate_classwise_density(discrete_scores, grid)

    pos_calibrated_hist = approximate_calibrated_density(classwise_density=belief_density,
                                                         grid=grid,
                                                         y=y,
                                                         k=k,
                                                         pos_case=True)

    neg_calibrated_hist = approximate_calibrated_density(classwise_density=belief_density,
                                                         grid=np.flip(grid),
                                                         y=y,
                                                         k=k,
                                                         pos_case=False)
    pos_scores = discrete_scores[y == k]
    neg_scores = discrete_scores[y != k]

    pos_uncalibrated_density = estimate_classwise_density(pos_scores, grid)
    neg_uncalibrated_density = estimate_classwise_density(neg_scores, grid)

    ot_pos_plan = compute_kantorovich_plan(pos_uncalibrated_density, pos_calibrated_hist)
    ot_neg_plan = compute_kantorovich_plan(neg_uncalibrated_density, neg_calibrated_hist)

    return ot_neg_plan, ot_pos_plan
def wasserstein_calibrated_classwise(pred, y, k, eta=2):
    discrete_scores, grid = discretize_scores(pred[:, k], eta=eta)
    belief_density = estimate_classwise_density(discrete_scores, grid)

    pos_calibrated_hist = approximate_calibrated_density(classwise_density=belief_density,
                                                     grid=grid,
                                                     y=y,
                                                     k=k,
                                                     pos_case=True)

    neg_calibrated_hist = approximate_calibrated_density(classwise_density=belief_density,
                                                     grid=np.flip(grid),
                                                     y=y,
                                                     k=k,
                                                     pos_case=False)
    pos_scores = discrete_scores[y == k]
    neg_scores = discrete_scores[y != k]

    pos_uncalibrated_density = estimate_classwise_density(pos_scores, grid)
    neg_uncalibrated_density = estimate_classwise_density(neg_scores, grid)

    ot_pos_plan = compute_kantorovich_plan(pos_uncalibrated_density, pos_calibrated_hist)
    ot_neg_plan = compute_kantorovich_plan(neg_uncalibrated_density, neg_calibrated_hist)

    pos_transformed_scores = apply_transport_plan(pos_scores, grid, ot_pos_plan)
    neg_transformed_scores = apply_transport_plan(neg_scores, grid, ot_neg_plan)

    transformed_scores = discrete_scores.copy()

    transformed_scores[y == k] = pos_transformed_scores
    transformed_scores[y != k] = neg_transformed_scores

    return transformed_scores


def apply_kwise_transport_plan(prob, y, k, eta, ot_plan):
    discrete_scores, grid = discretize_scores(prob[:, k], eta=eta)

    pos_scores = discrete_scores[y == k]
    neg_scores = discrete_scores[y != k]

    ot_pos_plan = ot_plan[1]
    ot_neg_plan = ot_plan[0]

    pos_transformed_scores = apply_transport_plan(pos_scores, grid, ot_pos_plan)
    neg_transformed_scores = apply_transport_plan(neg_scores, grid, ot_neg_plan)

    transformed_scores = discrete_scores.copy()

    transformed_scores[y == k] = pos_transformed_scores
    transformed_scores[y != k] = neg_transformed_scores

    return transformed_scores


def apply_all_ot_plans(prob, y, eta, ot_plans):
    K = np.unique(y)
    transformed_scores = [apply_kwise_transport_plan(prob=prob,
                                                     y=y,
                                                     k=k,
                                                     eta=eta,
                                                     ot_plan=ot_plans[k]
                                                    ).reshape(-1, 1) for k in K]

    new_vector = np.hstack(transformed_scores)
    return new_vector / new_vector.sum(axis=1).reshape(-1, 1)


class WassersteinCalibration:
    def __init__(self):
        self.ot_plans = None
    def fit(self, prob, y, eta=3):
        K = prob.shape[1]

        self.ot_plans = {}
        for k in range(K):
            neg_plan, pos_plan = transport_plan_k_calibrated(prob, y, k, eta=eta)
            self.ot_plans[k] = [neg_plan, pos_plan]

        transformed = apply_all_ot_plans(prob, y, eta, self.ot_plans)
        ece_before = calibration_error(prob, y)
        ece_after = calibration_error(transformed, y)

        print('Training ECE - Before %.3f, After: %.3f' % (ece_before, ece_after))


    def calibrate(self, prob, y, eta=3):
        assert self.ot_plans is not None


        transformed = apply_all_ot_plans(prob, y, eta, self.ot_plans)

        return transformed










