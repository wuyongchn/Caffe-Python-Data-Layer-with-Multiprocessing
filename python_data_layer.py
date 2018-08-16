import scipy.misc
import numpy as np
import random
import os
from PIL import Image
import caffe
from multiprocessing import Process, Queue
kQueueMaxBuffer = 10


class PythonDataLayer(caffe.Layer):
    """Python data layer used for training"""

    def setup(self, bottom, top):
        """Setup the PythonDataLayer"""
        self._top_names = ['data', 'label']
        params = eval(self.param_str)
        check_params(params)
        self._batch_size = params['batch_size']
        self._use_prefetch = params['use_prefetch']
        self._data_loader = NetworkDataLoader(params)

        top[0].reshape(self._batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self._batch_size, 1)

    def forward(self, bottom, top):
        im_blob, label_blob = self._data_loader.load_next_batch_blob()
        top[0].data[...] = im_blob
        top[1].data[...] = label_blob

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class NetworkDataLoader(object):
    def __init__(self, params):
        if 'use_prefetch' in params:
            self._use_prefetch = params['use_prefetch']
        else:
            self._use_prefetch = False

        if self._use_prefetch:
            self._queue = Queue(kQueueMaxBuffer)
            self._prefetch_process = BatchBlobFetcher(
                queue=self._queue, params=params, im_transformer=SimpleTransformer())
            self._prefetch_process.daemon = True
            self._prefetch_process.start()

            # Terminate the child process when the parent exits
            def cleanup():
                print "Terminating BlobFetcher"
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)
        else:
            self._batch_size = params['batch_size']
            self._root_folder = params['root_folder']
            self._im_shape = params['im_shape']
            self._source_list = [line.rstrip('\n') for line in open(
                params['source'])]
            if self._source_list[-1] == '':
                self._source_list.pop()
            self._cur = 0
            self._transformer = SimpleTransformer()

            self._blob_im = np.zeros(
                shape=(self._batch_size, 3, self._im_shape[0], self._im_shape[1]), dtype=np.float32)
            self._blob_label = np.zeros(
                shape=(self._batch_size, 1), dtype=np.int32)
        print "NetworkDataLoader initialized with {} images".format(len([line.rstrip('\n') for line in open(
                params['source'])]))

    def load_next_image(self):
        assert not self._use_prefetch, "You turned 'use_prefetch' on and can not call the function: 'load_next_image'"
        if self._cur == len(self._source_list):
            self._shuffle_source_list()
        # load an image
        image_name, label = self._source_list[self._cur].split(' ')
        self._cur += 1
        image_path = os.path.join(self._root_folder, image_name)
        assert os.path.exists(image_path), "the image does not exist: %s" % image_path
        im = np.asarray(Image.open(os.path.join(self._root_folder, image_name)))
        im = scipy.misc.imresize(im, self._im_shape)

        # do a simple horozontal flop as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # load and prepare ground truth
        label = np.asarray(np.int32(id))
        return self._transformer.preprocess(im), label

    def load_next_batch_blob(self):
        if self._use_prefetch:
            return self._queue.get()
        else:
            if self._cur + self._batch_size >= len(self._source_list):
                self._shuffle_source_list()
            for itt in range(self._batch_size):
                transformed_im, label = self.load_next_image()
                self._blob_im[itt, ...] = transformed_im
                self._blob_label[itt, ...] = np.int32(label)
            return self._blob_im, self._blob_label

    def _shuffle_source_list(self):
        """Randomly Shuffle the training data"""
        assert not self._use_prefetch, "You turned 'use_prefetch' on, do not need to shuffle source_list"
        self._cur = 0
        random.shuffle(self._source_list)


class BatchBlobFetcher(Process):
    def __init__(self, queue, params, im_transformer):
        super(BatchBlobFetcher, self).__init__()
        self._queue = queue
        self._batch_size = params['batch_size']
        self._source_list = [line.rstrip('\n') for line in open(
            params['source'])]
        if self._source_list[-1] == '':
            self._source_list.pop()
        self._root_folder = params['root_folder']
        self._im_shape = params['im_shape']
        self._cur = 0
        self._transformer = im_transformer
        self._shuffle_source_list()

        self._blob_im = np.zeros(
            shape=(self._batch_size, 3, self._im_shape[0], self._im_shape[1]), dtype=np.float32)
        self._blob_label = np.zeros(
            shape=(self._batch_size, 1), dtype=np.int32)

    def run(self):
        print "BlobFetcher Started"
        while True:
            if self._cur + self._batch_size >= len(self._source_list):
                self._shuffle_source_list()
            for itt in range(self._batch_size):
                image_name, label = self._source_list[self._cur].split(' ')
                self._cur += 1
                image_path = os.path.join(self._root_folder, image_name)
                assert os.path.exists(image_path), "the image does not exist: %s" % image_path
                im = np.asarray(Image.open(
                    os.path.join(self._root_folder, image_name)
                ))
                im = scipy.misc.imresize(im, self._im_shape)
                # do a simple horozontal flop as data augmentation
                flip = np.random.choice(2) * 2 - 1
                im = im[:, ::flip, :]
                transformed_im = self._transformer.preprocess(im)
                self._blob_im[itt, ...] = transformed_im
                self._blob_label[itt, ...] = np.int32(id)
            self._queue.put((self._blob_im, self._blob_label))

    def _shuffle_source_list(self):
        """Randomly Shuffle the training data"""
        self._cur = 0
        random.shuffle(self._source_list)


def check_params(params):
    required = ['batch_size', 'source', 'root_folder', 'im_shape']
    for r in required:
        assert r in params.keys(), "Params must include {}".format(r)


def print_info(name, params):
    """Output some info regarding the class"""
    print "{} initialized for source: {}, root_folder: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['source'],
        params['root_folder'],
        params['batch_size'],
        params['im_shape']
    )
    

class SimpleTransformer(object):
    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe
    """
    def __init__(self, mean=(103.939, 116.779, 123.68)):  # (b_mean, g_mean, r_mean)
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0  #(1.0 / 127.5)

    def set_mean(self, mean):
        self.mean = mean

    def set_scale(self, scale):
        self.scale = scale

    def preprocess(self, im):
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose(2, 0, 1)
        return im

    def deprocess(self, im):
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB
        return np.uint8(im)
