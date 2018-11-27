# Implementation of Shiga and Mamitsuka,
# IEEE Transactions on Knowledge and Data Engineering,
# Vol.24, no.4, april 2012

import numpy

from scipy.special import digamma
from scipy.special import gamma
from scipy.special import loggamma as complex_loggamma


def loggamma(*args):
    return complex_loggamma(*args).real


class Generative(object):
    def __init__(self, matrices, K):
        self.K = K
        self.M = len(matrices)
        self.V = len(matrices[0])

        self.weights = matrices

        # K = number of clusters
        # V = number of nodes
        # M = number of graphs

        # self.gamma = numpy.random.random((self.M, self.V, self.V, self.K))
        # self.z = numpy.random.random((self.M, self.V, self.V, self.K))
        self.gamma = numpy.zeros((self.M, self.V, self.V, self.K))
        self.z = numpy.zeros((self.M, self.V, self.V, self.K))

        # self.zeta = numpy.zeros((self.M, self.K))
        # self.beta = numpy.zeros((self.V, self.K))

        self.N_k = numpy.random.random((self.K))
        self.N_mk = numpy.random.random((self.M, self.K))
        self.N_ik = numpy.random.random((self.V, self.K))

        self.alpha_0 = 1.0
        self.beta_0 = 1.0
        self.zeta_0 = 1.0

        self.objective_values = []


    def z_avg(self):
        # print self.z
        results = numpy.zeros((self.V, self.K))
        for i in xrange(self.V):
            for k in xrange(self.K):
                z_vector = self.z[:, :, i, k]
                results[i, k] = numpy.mean(z_vector)
                # z_vector = self.z[0, :, i, k]
                # results[i, k] = numpy.mean(z_vector)

        for i in xrange(self.V):
            results[i, :] = results[i, :] / numpy.sum(results[i, :])

        return results


    def update_e(self):
        digamma_sum_alpha = digamma(numpy.sum(self.alpha))
        for i in xrange(self.V):
            for j in xrange(i + 1):
                for k in xrange(self.K):
                    zeta_k = self.zeta[:, k]
                    digamma_sum_zeta = digamma(numpy.sum(zeta_k))
                    beta_k = self.beta[:, k]
                    digamma_sum_beta = digamma(numpy.sum(beta_k))
                    for m in xrange(self.M):
                        if self.weights[m, i, j] > 0:
                            gamma_mijk = \
                                digamma(self.alpha[k]) \
                                - digamma_sum_alpha \
                                + digamma(zeta_k[m]) \
                                - digamma_sum_zeta \
                                + digamma(beta_k[i]) \
                                + digamma(beta_k[j]) \
                                - 2 * digamma_sum_beta
                            self.gamma[m, i, j, k] = gamma_mijk
                            # self.gamma[m, j, i, k] = gamma_mijk

        for i in xrange(self.V):
            for j in xrange(i + 1):
                for m in xrange(self.M):
                    if self.weights[m, i, j] > 0:
                        gamma_exp = [numpy.exp(x) for x in self.gamma[m, i, j, :]]
                        gamma_sum = numpy.sum(gamma_exp)
                        for k in xrange(self.K):
                            q_mijk = gamma_exp[k] / gamma_sum
                            self.z[m, i, j, k] = q_mijk
                            self.z[m, j, i, k] = q_mijk

    def update_m(self):
        # for k in xrange(self.K):
        #     value = 0.0
        #     for m in xrange(self.M):
        #         for i in xrange(self.V):
        #             for j in xrange(i):
        #                 if self.weights[m, i, j] > 0:
        #                     value += self.z[m, i, j, k] * self.weights[m, i, j]
        #     self.N_k[k] = value

        for k in xrange(self.K):
            for m in xrange(self.M):
                value = 0.0
                for i in xrange(self.V):
                    for j in xrange(i + 1):
                        if self.weights[m, i, j] > 0:
                            value += self.z[m, i, j, k] * self.weights[m, i, j]
                self.N_mk[m, k] = value

        self.N_k = self.N_mk.sum(axis=0)

        for k in xrange(self.K):
            for i in xrange(self.V):
                value = 0.0
                for m in xrange(self.M):
                    for j in xrange(self.V):
                        # if j != i and self.weights[m, i, j] > 0:
                        if self.weights[m, i, j] > 0:
                            value += self.z[m, i, j, k] * self.weights[m, i, j]
                self.N_ik[i, k] = value

    @property
    def alpha(self):
        return self.N_k + self.alpha_0

    @property
    def zeta(self):
        return self.N_mk + self.zeta_0

    @property
    def beta(self):
        return 2 * self.N_ik + self.beta_0

    def L(self):
        z_term = 0.0
        for k in xrange(self.K):
            for m in xrange(self.M):
                for i in xrange(self.V):
                    for j in xrange(i + 1):
                        if self.weights[m, i, j] > 0:
                            z_term += self.z[m, i, j, k] * numpy.log(self.z[m, i, j, k])

        L = loggamma(self.K * self.alpha_0) \
            - self.K * loggamma(self.alpha_0) \
            + self.K * loggamma(self.V * self.beta_0) \
            - self.K * self.V * loggamma(self.beta_0) \
            + self.K * loggamma(self.M * self.zeta_0) \
            - self.K * self.M * loggamma(self.zeta_0) \
            - z_term \
            - loggamma(numpy.sum(self.alpha)) \
            + numpy.sum([loggamma(x) for x in self.alpha]) \
            - numpy.sum([loggamma(x) for x in self.beta.sum(axis=0)]) \
            + loggamma(self.beta).sum() \
            - numpy.sum([loggamma(x) for x in self.zeta.sum(axis=0)]) \
            + loggamma(self.zeta).sum()

        self.objective_values.append(L)

        return L

    def pi(self, k):
        return self.alpha[k] / numpy.sum(self.alpha)

    def r(self, i, k):
        return self.beta[i, k] / numpy.sum(self.beta[:, k])

    def eta(self, m, k):
        return self.zeta[m, k] / numpy.sum(self.zeta[:, k])

    def prob(self, i, k):
        a1 = self.pi(k) * self.r(i, k) / numpy.sum([self.pi(kprime) * self.r(i, kprime) for kprime in xrange(self.K)])

        nom = numpy.sum([self.pi(k) * self.eta(m, k) * self.r(i, k) for m in xrange(self.M)]) 
        denom = 0
        for kprime in xrange(self.K):
            denom += numpy.sum([self.pi(kprime) * self.eta(m, kprime) * self.r(i, kprime) for m in xrange(self.M)])

        a2 = nom / denom
        return a2

    def prob_matrix(self):
        prob_output = numpy.zeros((self.V, self.K), dtype='float')
        for i in xrange(self.V):
            for k in xrange(self.K):
                prob_output[i, k] = self.prob(i, k)
        return prob_output

    def fit(self):
        iter_threshold = 5
        max_iter = 10000
        delta_threshold = 1e-13

        last = None
        deltas = []

        counter = 0
        while len(deltas) < iter_threshold or numpy.max(deltas) > delta_threshold:
            if counter > max_iter:
                break
            counter += 1
            self.update_e()
            self.update_m()
            current = self.L()
            if last:
                print current - last
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
                      [0, 0, 1, 1]],

                     [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]],

                     [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]],

                     [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]]])

    r = Generative(a, 3)
    r.fit()
    print r.prob_matrix()


def test2():
    # most definitely a bug, b/c consistently lower objective function...
    a = numpy.array([[[1, 1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0, 1],
                      [1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1]],

                     [[1, 1, 0, 1, 0, 0],
                      [1, 1, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0, 1],
                      [1, 0, 0, 1, 1, 1],
                      [0, 1, 0, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1]],

                     [[1, 0, 1, 0, 0, 1],
                      [0, 1, 0, 1, 0, 0],
                      [1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 0, 1]]], dtype='float')

    r = Generative(a, 2)
    r.fit()
    print r.prob_matrix()

    # r = Generative(numpy.array([a[0] + a[1] + a[2]]), 2)
    # r.fit()
    # print r.prob_matrix()


def test3():
    a = numpy.array([[[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]],

                     [[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]]])

    r = Generative(a, 2)
    r.fit()
    print r.prob_matrix()


def test4():
    # most definitely a bug, b/c consistently lower objective function...
    a = numpy.array([[[1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]],

                     [[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1]]], dtype='float')

    a = numpy.array([a[0], a[1], numpy.random.random((6, 6))])

    r = Generative(a, 4)
    r.fit()
    print r.prob_matrix()

    # r = Generative(numpy.array([a[0] + a[1] + a[2]]), 2)


def test5():
    # most definitely a bug, b/c consistently lower objective function...
    a = numpy.array([[[1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1]]], dtype='float')

    r = Generative(a, 2)
    r.fit()
    print r.prob_matrix()
    print r.z_avg()

    # r = Generative(numpy.array([a[0] + a[1] + a[2]]), 2)


def test6():
    # num_classes = 5
    # n_samp = 200
    num_classes = 3
    n_samp = 60
    labels = []
    for i in xrange(n_samp):
        labels.append(numpy.random.randint(num_classes))
    labels = numpy.array(labels)

    a = numpy.zeros((n_samp, n_samp))
    b = numpy.zeros((n_samp, n_samp))
    label_all = []
    for i in xrange(num_classes):
        l_c = numpy.where(labels == i)[0]
        label_all.append(l_c)

    for l in label_all:
        for i in l:
            for j in l:
                wt = 0.5 + 0.5 * numpy.random.random()  # 1
                a[i, j] = wt
                a[j, i] = wt
                wt = 0.5 + 0.5 * numpy.random.random()  # 1
                b[i, j] = wt
                b[j, i] = wt

    r = Generative(numpy.array([a, b]), num_classes)
    r.fit()

    # print r.prob_matrix()
    # print r.z_avg()
    p_m = numpy.argmax(r.prob_matrix(), axis=1)
    z_m = numpy.argmax(r.z_avg(), axis=1)
    # labels
    pred_mismatch = 0
    label_error = 0
    label_map = {}
    for a, b, c in zip(p_m, z_m, labels):
        # print a, b, c
        if a != b:
            pred_mismatch += 1
        if a in label_map:
            if c != label_map[a]:
                label_error += 1
        else:
            label_map[a] = c

    print 'Label mismatch: %s' % label_error
    print 'Prediction mismatch: %s' % pred_mismatch

        
def test7():
    num_classes = 3
    num_samples = 300

    labels = []
    for i in xrange(num_samples):
        labels.append(numpy.random.randint(num_classes))
    labels = numpy.array(labels)

    num_samples = 9
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    a = numpy.zeros((num_classes, num_samples, num_samples))
    
    for i in xrange(num_samples):
        for j in xrange(i + 1):
            for l in xrange(num_classes):
                if labels[i] == l and labels[j] == l:
                    a[l, i, j] = 1
                    a[l, j, i] = 1
                if labels[i] != l and labels[j] != l:
                    a[l, i, j] = 1
                    a[l, j, i] = 1

    print a

    r = Generative(a, num_classes)
    r.fit()

    p_m = numpy.argmax(r.prob_matrix(), axis=1)
    z_m = numpy.argmax(r.z_avg(), axis=1)

    print p_m
    print z_m

    print r.prob_matrix()
    print r.z_avg()

    pred_mismatch = 0
    label_error = 0
    label_map = {}
    for a, b, c in zip(p_m, z_m, labels):
        if a != b:
            pred_mismatch += 1
        if a in label_map:
            if c != label_map[a]:
                label_error += 1
        else:
            label_map[a] = c

    print 'Label mismatch: %s' % label_error
    print 'Prediction mismatch: %s' % pred_mismatch
            

def test8():
    a = numpy.zeros((1, 15, 15))
    a[:, :5, :5] = 1
    a[:, 5:, 5:] = 1
    a[:, 2, 7] = 1
    a[:, 7, 2] = 1

    r = Generative(a, 2)
    r.fit()

    print r.prob_matrix()


def test9():
    n_samp = 300
    n_cluster = 3
    centroids = ((0, 0), (0, 1), (1, 0))

    samples = []
    for c in xrange(n_cluster):
        for i in xrange(n_samp / n_cluster):
            p = numpy.random.randn(2) * 0.1 + centroids[c]
            samples.append(p)

    a = numpy.zeros((3, n_samp, n_samp))
    for c in xrange(n_cluster):
        for i in xrange(n_samp):
            for j in xrange(i):
                lower = c * (n_samp / n_cluster)
                upper = (c + 1) * (n_samp / n_cluster)
                limit = 0.1 if lower <= i < upper  and lower <= j < upper else 0.9
                
                print lower, upper, i, j, limit
                edge = 1 if numpy.random.random() > limit else 0
                a[c, i, j] = edge
                a[c, j, i] = edge

    r = Generative(a, n_cluster)
    r.fit()

    print r.prob_matrix()
    print r.z_avg()

    print numpy.argmax(r.prob_matrix(), axis=1)
