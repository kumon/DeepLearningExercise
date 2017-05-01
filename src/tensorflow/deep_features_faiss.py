import numpy as np
import faiss
import time
import json

def main():
    with open('data.json', 'r') as f:
        xb = np.array([np.array(json.loads(l.strip())).astype('float32') for l in f])

    for i in range(len(xb)):
        xb[i][0] += np.random.random()

    nq = 1
    xq = np.copy(xb[:nq])
    nb, d = xb.shape
    n_candidates = 10

    # Search (brute force)
    s = time.time()
    result_d, result_i = [], []
    for q in xq:
        dist = np.array([np.linalg.norm(d) for d in (xb - q)])
        idx = np.array(sorted(range(len(dist)), key=lambda k: dist[k])[:n_candidates])
        result_d.append(dist[idx])
        result_i.append(idx)
    result_d, result_i = np.array(result_d), np.array(result_i)
    print('Average query time (brute force): {:.2f} [ms]'.format((time.time() - s) * 1000 / nq))


    # Index (faiss)
    s = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    print('Index time (faiss): {:.2f} [ms]'.format((time.time() - s) * 1000))

    # Search (faiss)
    s = time.time()
    result_d1, result_i1 = index.search(xq, n_candidates)
    print('Average query time (faiss): {:.2f} [ms]'.format((time.time() - s) * 1000 / nq))

    # Evaluate (faiss)
    evaluate(result_i, result_i1)

    # Index (faiss (quantize))
    s = time.time()
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)
    print('Index time (faiss / quantize): {:.2f} [ms]'.format((time.time() - s) * 1000))

    # Search (faiss (quantize nprobe=1))
    s = time.time()
    index.nprobe = 1
    result_d2, result_i2 = index.search(xq, n_candidates)
    print('Average query time (faiss / quantize nprobe=1): {:.2f} [ms]'.format((time.time() - s) * 1000 / nq))

    # Evaluate (faiss (quantize nprobe=1))
    evaluate(result_i, result_i2)

    # Search (faiss (quantize nprobe=10))
    s = time.time()
    index.nprobe = 10
    result_d3, result_i3 = index.search(xq, n_candidates)
    print('Average query time (faiss / quantize nprobe=10): {:.2f} [ms]'.format((time.time() - s) * 1000 / nq))

    # Evaluate (faiss (quantize nprobe=10))
    evaluate(result_i, result_i3)


# Evaluate
def evaluate(arr1, arr2):
    top_1 = (arr1[:,0] == arr2[:,0]).sum() / arr1.shape[0]
    total = 0
    for t in np.c_[arr1, arr2]:
        _, cnt = np.unique(t, return_counts=True)
        total += (cnt >= 2).sum()
    top_k = total / arr1.shape[0] / arr1.shape[1]
    print('recall@1: {:.2f}, top {} recall: {:.2f}'.format(top_1, arr1.shape[1], top_k))


if __name__ == "__main__":
    main()

