from helper import *

IMG_DIR = '/path/to/img'
MODEL_PATH = '/path/to/classify_image_graph_def.pb'
IMG_NUM = 1408
QUERY_IMG = 22
CANDIDATES = 5

with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    features = []
    for i in range(IMG_NUM):
        image_data = tf.gfile.FastGFile('%s/img_%04d.jpg' % (IMG_DIR, i), 'rb').read()
        pool3_features = sess.run(pool3,{'DecodeJpeg/contents:0': image_data})
        features.append(np.squeeze(pool3_features))

query_feat = features[QUERY_IMG]
sims = [(k, round(1 - spatial.distance.cosine(query_feat, v), 3)) for k,v in enumerate(features)]
print(sorted(sims, key=operator.itemgetter(1), reverse=True)[:CANDIDATES + 1])

