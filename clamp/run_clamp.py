import csv
import numpy

import clamp3


def cluster(filename, K):
    # Read input file and run clustering
    raw_input_data = []
    with open(filename) as f:
        r = csv.reader(f, delimiter='\t')
        # skip header line
        r.next()
        for l in r:
            raw_input_data.append(l)

    # filter clusters
    cluster_names = set([])
    for line in raw_input_data:
        cluster_names.add(line[0])
        cluster_names.add(line[1])

    # choose random cluster names to skip
    skip_ratio = 0.25
    # skip_ratio = 0.0
    skip_size = int(len(cluster_names) * skip_ratio)
    if skip_size > 0:
        cluster_names_to_skip = numpy.random.choice(list(cluster_names), skip_size, replace=False)
    else:
        cluster_names_to_skip = []

    filtered_input_data = []
    for line in raw_input_data:
        if not (line[0] in cluster_names_to_skip or line[1] in cluster_names_to_skip):
            filtered_input_data.append(line)

    # K
    family_names = set([])
    with open('/home/grimur/data-crusemann/bigscape-crusemann-alt/network_files/2018-06-21_15-33-41_glocal/PKSother/PKSother_clustering_c0.55.tsv') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            name, fam = line.strip().split()
            if name not in cluster_names_to_skip:
                family_names.add(fam)

    raw_input_data = filtered_input_data
    K = len(family_names)

    input_len = len(raw_input_data)
    num_nodes = (numpy.sqrt(1 + 8 * input_len) - 1) / 2 + 1
    num_nodes = int(num_nodes)
    distance_columns = [4, 5, 6]
    edges = numpy.zeros((num_nodes, num_nodes, len(distance_columns)))

    cluster_indices = []
    for line in raw_input_data:
        cluster_a = line[0]
        cluster_b = line[1]

        if cluster_a in cluster_indices:
            cluster_a_idx = cluster_indices.index(cluster_a)
        else:
            cluster_a_idx = len(cluster_indices)
            cluster_indices.append(cluster_a)
        if cluster_b in cluster_indices:
            cluster_b_idx = cluster_indices.index(cluster_b)
        else:
            cluster_b_idx = len(cluster_indices)
            cluster_indices.append(cluster_b)

        for distance_idx, distance_column_idx in enumerate(distance_columns):
            d = float(line[distance_column_idx])
            wt = 1 / numpy.exp(d)
            edges[cluster_a_idx, cluster_b_idx, distance_idx] = wt
            edges[cluster_b_idx, cluster_a_idx, distance_idx] = wt

    edge_list = [edges[:, :, x] for x in xrange(len(distance_columns))]

    # PKSI: 68
    # PKSother: 39
    # PKS-NRP_Hybrids: 33
    # RiPPs: 75
    # Saccharides: 53
    # Terpene: 34
    # NRPS: 118

    # ('NRPS', [2.5662555196840572e-07, 0.99990785713116881, 9.1886243279235618e-05])
    # ('PKSI', [2.8263101967528905e-07, 0.99995914393075436, 4.0573438225938243e-05])
    # ('Saccharides', [0.98006109263543972, 0.010151473390183892, 0.0097874339743763122])
    # ('PKSother', [0.54250959512296693, 1.747449235248119e-08, 0.45749038740254067])
    # ('PKS-NRP_Hybrids', [0.87449637927294455, 0.0051778761690297985, 0.12032574455802564])
    # ('RiPPs', [0.029615490363663493, 0.0051539361031855283, 0.96523057353315089])
    # ('Terpene', [0.51803204158501925, 8.0196499660023943e-05, 0.48188776191532079])

    # c = clamp3.CLAMP(edge_list, numpy.array([[]]), K,  1.2, 5)
    # c.fit([0.99, 0.01])
    c = clamp3.CLAMP(edge_list, numpy.array([[]]), K,  1.1, 1000)
    c.fit([0.99, 0.01])
    # c = clamp3.CLAMP(edge_list, numpy.array([[]]), K,  1.5, 3)
    # c.fit([0.99, 0.01])
    # c = clamp3.CLAMP(edge_list, numpy.array([[]]), K,  1.2, 1.2)
    # c.fit([0.99, 0.01])
    # c = clamp3.CLAMP(edge_list, numpy.array([[]]), K,  1.5, 1.5)
    # c.fit([0.99, 0.01])

    # print numpy.argmax(c.cluster_probs, axis=1)
    return c.edge_wts, c.cluster_probs, cluster_indices
    # wss_pairs = []
    # for k in xrange(10, 150, 10):
    #     # k=118 from bigscape
    #     c = clamp.CLAMPoptimiser(edge_list, numpy.array([[]]), k,  1.5, 1.5)
    #     c.fit()
    #     wss = c.wss()
    #     print wss
    #     wss_pairs.append((k, wss))

    # for k, wss in wss_pairs:
    #     print k, wss


if __name__ == '__main__':
    root = '/home/grimur/data-crusemann/bigscape-crusemann-alt/network_files/2018-06-21_15-33-41_glocal/%s/%s_c0.55.network'
    pairs = [
            # ("NRPS", 118),
            # ("PKSI", 68),
            # ("Saccharides", 53),
            ("PKSother", 39),
            # ("PKS-NRP_Hybrids", 33),
            # ("RiPPs", 75),
            # ("Terpene", 34),
            ]
    edges = []
    probs = []
    labels = []
    for product_type, k in pairs:
      for i in xrange(3):
        filename = root % (product_type, product_type)
        e, p, l = cluster(filename, k)
        edges.append((product_type, e))
        probs.append((product_type, p))
        labels.append((product_type, l))
    # cluster('/home/grimur/data-crusemann/bigscape-crusemann-alt/network_files/2018-06-21_15-33-41_glocal/NRPS/NRPS_c0.55.network')
    # cluster('/home/grimur/data-crusemann/bigscape-crusemann-alt/network_files/2018-06-21_15-33-41_glocal/PKSI/PKSI_c0.55.network')
    # cluster('/home/grimur/data-crusemann/bigscape-crusemann-alt/network_files/2018-06-21_15-33-41_glocal/PKSother/PKSother_c0.55.network')
    for e in edges:
        print e
    with open('probs-saccharides-all-medf3.bin', 'wb') as f:
        import cPickle
        cPickle.dump((edges, probs, labels), f)
