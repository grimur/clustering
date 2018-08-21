# Implementation of Shiga and Mamitsuka,
# IEEE Transactions on Knowledge and Data Engineering,
# Vol.24, no.4, april 2012

import numpy

from scipy.special import digamma
from scipy.special import gamma


class Generative(object):
    def __init__(self, matrices, K):
        self.K = K
        self.M = len(matrices)
        self.V = len(matrices[0])

        self.weights = matrices

        # K = number of clusters
        # V = number of nodes
        # M = number of graphs
        self.gamma = numpy.random.random((self.M, self.V, self.V, self.K))
        self.z = numpy.random.random((self.M, self.V, self.V, self.K))

        self.alpha = numpy.random.random((self.K))
        self.zeta = numpy.random.random((self.M, self.K))
        self.beta = numpy.random.random((self.V, self.K))

        # Used in objective function
        self.q_pi = 0
        self.q_eta = numpy.zeros((self.K))
        self.q_r = numpy.zeros((self.K))

        # These need to be properly initialised?
        # see section 4.3, footnote 4

        # self.alpha_0 = numpy.ones((self.K))
        # self.zeta_0 = numpy.ones((self.M, self.K))
        # self.beta_0 = numpy.ones((self.V, self.K))
        self.alpha_0 = 1.0
        self.beta_0 = 1.0
        self.zeta_0 = 1.0

    def update_e(self):
        # update q(z^m_ij = k) -- eq. 17
        # m-step updates alpha, zeta, beta.

        # this is updating gamma, see post-eq. 17
        # TODO: Optimise. This is symmetric in i and j
        for m in xrange(self.M):
            for i in xrange(self.V):
                for j in xrange(self.V):
                    for k in xrange(self.K):
                        t1 = digamma(self.alpha[k])
                        t2 = digamma(numpy.sum(self.alpha))
                        t3 = digamma(self.zeta[m, k])
                        t4 = digamma(numpy.sum(self.zeta[:, k]))
                        t5 = digamma(self.beta[i, k])
                        t6 = digamma(self.beta[j, k])
                        t7 = 2 * digamma(numpy.sum(self.beta[:, k]))
                        self.gamma[m, i, j, k] = t1 - t2 + t3 - t4 + t5 + t6 - t7

        # this is eqn 17
        # TODO: Optimise. This is symmetric in i and j
        for m in xrange(self.M):
            for i in xrange(self.V):
                for j in xrange(self.V):
                    gamma_vector = self.gamma[m, i, j, :]
                    gamma_exp = [numpy.exp(x) for x in gamma_vector]
                    for k in xrange(self.K):
                        self.z[m, i, j, k] = gamma_exp[k] / numpy.sum(gamma_exp)

    def update_m(self):
        # eqns 11 - 16
        # update alpha_k -- number of clusters

        # eq 11
        N_k = numpy.zeros((self.K))
        for k in xrange(self.K):
            value = 0
            for m in xrange(self.M):
                for i in xrange(self.V):
                    for j in xrange(i):
                        value += self.z[m, i, j, k] * self.weights[m, i, j]
            N_k[k] = value

        # eq 12
        N_mk = numpy.zeros((self.M, self.K))
        for k in xrange(self.K):
            for m in xrange(self.M):
                value = 0
                for i in xrange(self.V):
                    for j in xrange(i):
                        value += self.z[m, i, j, k] * self.weights[m, i, j]
                N_mk[m, k] = value

        # eq 13
        N_ik = numpy.zeros((self.V, self.K))
        for k in xrange(self.K):
            for i in xrange(self.V):
                value = 0
                for m in xrange(self.M):
                    for j in xrange(self.V):
                        value += self.z[m, i, j, k] * self.weights[m, i, j]
                N_ik[i, k] = value

        # eq 14
        self.alpha = N_k + self.alpha_0
        self.q_pi = gamma(numpy.sum(self.alpha)) / \
            numpy.prod([gamma(x) for x in self.alpha]) * \
            numpy.prod([x**(y-1) for x, y in zip(self.pi, self.alpha)])

        # eq 15
        self.zeta = N_mk + self.zeta_0
        for k in xrange(self.K):
            self.q_eta[k] = gamma(numpy.sum(self.zeta[:, k])) / \
                numpy.prod([gamma(x) for x in self.gamma[:, k]]) * \
                numpy.prod([x**(y-1) for x, y in zip(self.eta[:, k], self.gamma[:, k])])

        # eq 16
        self.beta = 2 * N_ik + self.beta_0
        self.q_r = 0

    @property
    def pi(self):
        return self.alpha / self.alpha.sum()

    @property
    def eta(self):
        return self.zeta / self.zeta.sum(axis=0)

    @property
    def r(self):
        return self.beta / self.beta.sum(axis=0)

    def L(self):
        log = numpy.log
        K = self.K
        N = self.V
        M = self.M

        alpha_0 = self.alpha_0
        beta_0 = self.beta_0
        zeta_0 = self.zeta_0

        alpha = self.alpha
        beta = self.beta
        zeta = self.zeta

        selective_term = 0
        # TODO: This is basically summing over a matrix filtered
        # by another matrix. Surely numpy can do this quickly.
        for k in xrange(K):
            for m in xrange(M):
                for i in xrange(N):
                    for j in xrange(i):
                        if self.weights[m, i, j] != 0:
                            z = self.z[m, i, j, k]
                            selective_term += z * log(z)

        L = 0
        # first line
        L += log(gamma(K * alpha_0))
        L -= K * N * log(gamma(alpha_0))
        L += K * log(gamma(N * beta_0))
        # second line
        L -= K * N * log(gamma(beta_0))
        L += K * log(gamma(M * zeta_0))
        # third line
        L -= K * M * log(gamma(zeta_0))
        L -= selective_term
        # fourth line
        L -= log(gamma(sum(alpha)))
        L += sum([log(gamma(a)) for a in alpha])
        # fifth line
        L -= sum([log(gamma(x)) for x in beta.sum(axis=0)])
        L += numpy.sum(log(gamma(beta)))
        # sixth line
        L -= sum([log(gamma(x)) for x in zeta.sum(axis=0)])  # roughly same as line 5.1
        L += numpy.sum(log(gamma(zeta)))

        return L

    def prob(self, i, k):
        # Should there be a M term here?
        # This is adapting eqn 8 from paper, which is for a single graph!
        return self.pi[k] * self.r[i, k] / numpy.sum([p * r for p, r in zip(self.pi, self.r[i, :])])

    def prob_matrix(self):
        prob = numpy.zeros((self.V, self.K))

        # TODO: This can be optimised by caching in the prob function
        # (or doing all probabilities for a given vertex at once)
        for i in xrange(self.V):
            for k in xrange(self.K):
                p = self.prob(i, k)
                prob[i, k] = p

        return prob

    def fit(self):
        iter_threshold = 5
        max_iter = 10000
        delta_threshold = 1e-15

        last = None
        deltas = []

        counter = 0
        while len(deltas) < iter_threshold or numpy.max(deltas) > delta_threshold:
            if counter > max_iter:
                break
            counter += 1
            self.update_m()
            self.update_e()
            current = self.L()
            if last:
                delta = numpy.abs(last - current)
                deltas.append(delta)
            if len(deltas) > iter_threshold:
                deltas.pop(0)
            last = current


def test():
    a = numpy.array([[[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]],

                     [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]]])

    r = Generative(a, 2)
    r.fit()
    print r.prob_matrix()

