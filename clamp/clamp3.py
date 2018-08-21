import numpy


def attribute_distance(a, b):
    return 0


class CLAMP(object):
    def __init__(self, edges, attrs, K, f, lmbd):
        print 'Init clamp to %s clusters, f: %s, lmbd: %s' % (K, f, lmbd)
        self.edges = edges
        self.attrs = attrs
        self.K = K
        self.f = f
        self.lmbd = lmbd

        self.num_vertices = self.edges[0].shape[0]
        self.num_edge_types = len(self.edges)
        self.num_attr_types = self.attrs.shape[1]

        #######
        self.tot_attr_wt = 0.5
        self.tot_edge_wt = 0.5

        if self.num_edge_types > 0:
            self.edge_wts = [1.0 / self.num_edge_types] * self.num_edge_types
        else:
            self.edge_wts = []

        if self.num_attr_types > 0:
            self.attr_wts = [1.0 / self.num_attr_types] * self.num_attr_types
        else:
            self.attr_wts = []

        self.cluster_probs = numpy.random.random((self.num_vertices, self.K))

        for v in xrange(self.num_vertices):
            self.cluster_probs[v, :] = self.cluster_probs[v, :] / numpy.sum(self.cluster_probs[v, :])

        self.cluster_outgoing_edges = numpy.zeros((self.num_edge_types, self.num_vertices, self.K))
        self.cluster_attributes = numpy.zeros((self.num_attr_types, self.K))
        self.cluster_distances = numpy.zeros((self.num_vertices, self.K))

    def update_cluster_outgoing_edges(self):
        for t in xrange(self.num_edge_types):
            for u in xrange(self.num_vertices):
                for k in xrange(self.K):
                    prob_vector = self.cluster_probs[:, k]
                    dist_vector = self.edges[t][:, u]
                    prob_power_vector = numpy.power(prob_vector, self.f)
                    w = numpy.sum(numpy.dot(prob_power_vector, dist_vector))
                    self.cluster_outgoing_edges[t, u, k] = w / numpy.sum(prob_power_vector)

    def update_cluster_attributes(self):
        for a in xrange(self.num_attr_types):
            for k in xrange(self.K):
                prob_vector = self.cluster_probs[:, k]
                attr_vector = self.attrs[:, a]
                prob_power_vector = numpy.power(prob_vector, self.f)
                w = numpy.sum(numpy.dot(prob_power_vector, attr_vector))
                self.cluster_attributes[a, k] = w / numpy.sum(prob_power_vector)

    def update_cluster_distances(self):
        for k in xrange(self.K):
            for v in xrange(self.num_vertices):
                ad = self.cluster_attribute_distance(v, k)
                ed = self.cluster_edge_distance(v, k)
                self.cluster_distances[v, k] = self.tot_attr_wt * ad + self.tot_edge_wt * ed

    def cluster_attribute_distance(self, v, k):
        attr_sum = 0
        for a in xrange(self.num_attr_types):
            w_a = self.attr_wts[a]
            ad_a = self.cluster_type_attribute_distance(v, k, a)
            attr_sum = w_a * ad_a
        return attr_sum

    def cluster_type_attribute_distance(self, v, k, a):
        v_a = self.attrs[v, a]
        k_a = self.cluster_attributes[a, k]
        # prob_vector = self.cluster_probs[:, k]
        # dist_vector = self.attrs[:, a]
        # prob_power_vector = numpy.power(prob_vector, self.f)
        # w = numpy.sum(numpy.dot(prob_power_vector, dist_vector))
        # k_a = w / numpy.sum(prob_power_vector)
        return (v_a - k_a) ** 2

    def cluster_edge_distance(self, v, k):
        dist_sum = 0
        for t in xrange(self.num_edge_types):
            sc_t = self.cluster_type_edge_distance(v, k, t)
            w_t = self.edge_wts[t]
            dist_sum += w_t * sc_t
        return dist_sum / self.num_vertices

    def cluster_type_edge_distance(self, v, k, t):
        sc_t = 0
        for w in xrange(self.num_vertices):
            edge_1 = self.edges[t][v, w]
            # SAME AS CLUSTER_OUTGOING_EDGES[T, W, K]?
            # edge_2 = self.cluster_type_edge(w, k, t)
            edge_2 = self.cluster_outgoing_edges[t, w, k]
            sc_t += (edge_1 - edge_2) ** 2
        return sc_t

    def cluster_type_edge(self, w, k, t):
        k_probs = self.cluster_probs[:, k]
        k_probs_power = numpy.power(k_probs, self.f)
        w_vector = self.edges[t][:, w]
        weight_sum = numpy.sum(numpy.dot(k_probs_power, w_vector))
        prob_sum = numpy.sum(k_probs_power)
        return weight_sum / prob_sum

    def update_membership_probabilities(self):
        # update cluster distances
        # self.update_cluster_distances()

        for v in xrange(self.num_vertices):
            for k in xrange(self.K):
                v_dist_vector = self.cluster_distances[v, :]
                vk_dist = v_dist_vector[k]
                ratio_vector = numpy.power([vk_dist / vj_dist for vj_dist in v_dist_vector], 1 / (self.f - 1))
                self.cluster_probs[v, k] = 1.0 / numpy.sum(ratio_vector)

    def update_specific_weights(self):
        s_vector = []
        for t in xrange(self.num_edge_types):
            s_t = 0
            for v in xrange(self.num_vertices):
                for k in xrange(self.K):
                    prob = self.cluster_probs[v, k]
                    dist = self.cluster_type_edge_distance(v, k, t)
                    s_t += (prob ** self.f) * dist
            s_t = self.tot_edge_wt * s_t
            s_vector.append(s_t)

        s_t_sum = numpy.sum([numpy.exp((s_vector[t] * numpy.log(2)) / (-self.lmbd)) for t in xrange(self.num_edge_types)])
        for t in xrange(self.num_edge_types):
            s_t = s_vector[t]
            w_t = numpy.exp((s_t * numpy.log(2)) / (-self.lmbd)) / s_t_sum
            self.edge_wts[t] = w_t

        # similar for attr_weights
        ad_vector = []
        for a in xrange(self.num_attr_types):
            ad_a = 0
            for v in xrange(self.num_vertices):
                for k in xrange(self.K):
                    prob = self.cluster_probs[v, k]
                    dist = self.cluster_type_attribute_distance(v, k, a)
                    ad_a = (prob ** self.f) * dist
            ad_a = self.tot_attr_wt * ad_a
            ad_vector.append(ad_a)

        ad_a_sum = numpy.sum([numpy.exp((ad_vector[a] * numpy.log(2)) / (-self.lmbd)) for a in xrange(self.num_attr_types)])
        for a in xrange(self.num_attr_types):
            ad_a = ad_vector[a]
            w_a = numpy.exp((ad_a * numpy.log(2)) / (-self.lmbd)) / ad_a_sum
            self.attr_wts[a] = w_a

    def update_global_weights(self):
        # self.tot_attr_wt = 1.0
        # self.tot_edge_wt = 0.0
        # todo: IMPLEMENT THIS!!!
        pass

    def update(self):
        self.update_cluster_outgoing_edges()
        # print 'Updated cluster outgoing edges'
        self.update_cluster_attributes()
        # print 'Updated cluster attributes'
        self.update_cluster_distances()
        # print 'Updated cluster distances'
        self.update_membership_probabilities()
        # print 'Updated cluster membership probabilities'
        self.update_specific_weights()
        # print 'Updated specific weights'
        self.update_global_weights()
        # print 'Updated global weights'

    def objective(self):
        dist_term = 0
        for v in xrange(self.num_vertices):
            for k in xrange(self.K):
                prob = self.cluster_probs[v, k]
                dist = self.cluster_distances[v, k]
                dist_term += prob ** self.f * dist

        reg_term = 0
        if self.tot_attr_wt > 0:
            reg_term += self.tot_attr_wt * numpy.log(self.tot_attr_wt)
        if self.tot_edge_wt > 0:
            reg_term += self.tot_edge_wt * numpy.log(self.tot_edge_wt)
        reg_term += numpy.sum([a * numpy.log(a) for a in self.attr_wts])
        reg_term += numpy.sum([t * numpy.log(t) for t in self.edge_wts])

        return dist_term + self.lmbd * reg_term

    def display(self):
        print self.cluster_probs
        # print self.edge_wts
        # print self.objective()

    def display_all(self):
        print self.cluster_probs
        print self.edge_wts
        print self.attr_wts

    def fit(self, wt=None):
        self.initial = True
        if wt is None:
            self.tot_attr_wt = 0.5
            self.tot_edge_wt = 0.5
        else:
            self.tot_edge_wt, self.tot_attr_wt = wt

        objectives = [self.objective()]
        print 'Initialised'

        tail = 3
        threshold = 1e-50
        threshold = 1e-50
        for i in xrange(5000):
            self.update()
            # self.display()
            objective = self.objective()
            print 'iter: %s, obj: %s' % (i, objective)
            print self.edge_wts
            objectives.append(objective)

            # Only start monitoring the objective progress
            # once we have enough data points
            if len(objectives) < tail:
                continue

            last_objectives = objectives[-tail:]
            exit = True
            for t in xrange(tail - 1):
                if not numpy.abs(last_objectives[t] - last_objectives[t + 1]) < threshold:
                    exit = False
            if exit:
                print 'Exited after %s iterations' % i
                break





def test_1():
    wts_1 = numpy.array([[1, .9, .1, .1],
                         [.9, 1, .1, .1],
                         [.1, .1, 1, .9],
                         [.1, .1, .9, 1]])
    edges = [wts_1, numpy.random.random((4, 4))]

    attrs = numpy.array([[],
                         [],
                         [],
                         []])

    c = CLAMP(edges, attrs, 2, 1.1, 1)
    c.fit([1.0, 0.0])
    c.display_all()


def test_2():
    wts_1 = numpy.array([[1, .9, .1, .1, .1, .1],
                         [.9, 1, .1, .1, .1, .1],
                         [.1, .1, 1, .9, .1, .1],
                         [.1, .1, .9, 1, .1, .1],
                         [.1, .1, .1, .1, 1, .9],
                         [.1, .1, .1, .1, .9, 1]])

    edges = [wts_1, numpy.random.random((6, 6))]
    attrs = numpy.array([[],
                         [],
                         [],
                         [],
                         [],
                         []])

    c = CLAMP(edges, attrs, 3, 1.1, 1)
    c.fit([1.0, 0.0])
    c.display_all()


def test_3():
    N = 30
    x = numpy.random.normal(0, 1.0, (N, 2))
    x[10:20, :] += 3
    x[20:, :] += [-3, 3]
    W = numpy.zeros((N, N))
    for n in xrange(N):
        for m in xrange(N):
            if n == m:
                W[n, m] = 1.0
            else:
                W[n, m] = 1.0 / numpy.exp(-0.01 * (((x[n, :] - x[m, :])**2).sum()))

    randomW = numpy.random.rand(N, N)
    randomW = 0.5 * numpy.dot(randomW, randomW.T)
    for i in range(N):
        randomW[i, i] = 1.0

    edges = [randomW, W]
    attrs = numpy.array([[]] * N)

    c = CLAMP(edges, attrs, 3, 1.1, 1)
    c.fit([1.0, 0.0])
    c.display_all()


def test_attr_1():
    edges = [numpy.random.random((4, 4))]
    attrs = numpy.array([[1],
                         [1],
                         [0],
                         [0]])

    c = CLAMP(edges, attrs, 2, 1.1, 1)
    c.fit([0.0, 1.0])
    c.display_all()


def test_attr_2():
    wts_1 = numpy.array([[1, .9, .1, .1, .1, .1],
                         [.9, 1, .1, .1, .1, .1],
                         [.1, .1, 1, .9, .1, .1],
                         [.1, .1, .9, 1, .1, .1],
                         [.1, .1, .1, .1, 1, .9],
                         [.1, .1, .1, .1, .9, 1]])
    edges = [wts_1]
    attrs = numpy.array([[1, 1, 0],
                         [1, 2, 1],
                         [1, 3, 2],
                         [0, 1, 0],
                         [0, 2, 1],
                         [0, 3, 2]])

    c = CLAMP(edges, attrs, 2, 1.1, 1)
    c.fit([0.0, 1.0])
    c.display_all()


def test_attr_3():
    N = 30
    attr = numpy.random.normal(0, 1.0, (N, 1))
    attr2 = numpy.random.normal(0, 1.0, (N, 1)) * 25
    attr[10:20, 0] += 12
    attr[20:30, 0] -= 12
    attr = attr / numpy.std(attr)
    attr2 = attr2 / numpy.std(attr2)
    edges = [numpy.ones((N, N))]

    attr_comb = numpy.hstack((attr, attr2))

    c = CLAMP(edges, attr_comb, 3, 1.1, 1.5)
    c.fit([0.0, 1.0])
    c.display_all()



if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    # test_attr_1()
    # test_attr_2()
    # test_attr_3()

