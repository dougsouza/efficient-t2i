from pathlib import Path
import argparse
from PIL import Image
from inception.slim import slim
import numpy as np
import tensorflow as tf
import math
import scipy.misc 
import sys
import pickle
import glob
import h5py
import json


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=Path,
                        help='Path to inception checkpoint')
    parser.add_argument('--h5_file', type=Path, default=None,
                        help='Path h5 file containing images')
    parser.add_argument('--num_classes', type=int, default=50,
                        help='num of output classes (birds=50;flowers=20)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='pretty self explanatory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu ID to use')
    parser.add_argument('--splits', type=int, default=10,
                            help='number of splits')
    parser.add_argument('--gmodel', type=str, default=None,
                            help='name of generator model')
    return parser.parse_args()



# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)


def get_inception_score(sess, images, pred_op, batch_size, splits):
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        if i % 50 == 0:
            print("%d of %d batches" % (i, n_batches), flush=True),
        sys.stdout.flush()
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) -
              np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    # print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)



def build_inception_graph(input_placeholder, num_classes, for_training=False,
                                        restore_logits=True, scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
              input_placeholder,
              dropout_keep_prob=0.8,
              num_classes=num_classes,
              is_training=for_training,
              restore_logits=restore_logits,
              scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


def main(args):
    """Evaluate model on Dataset for a number of steps."""
    
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % args.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = args.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [args.batch_size, 299, 299, 3],
                    name='inputs')

                logits, _ = build_inception_graph(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = \
                    tf.slice(logits, [0, 1],
                             [args.batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, str(args.checkpoint_dir))
                print('Restore the model from %s).' % args.checkpoint_dir)


                h5_file = Path(args.h5_file)

                with h5py.File(h5_file, 'r') as h5file:
                    images = h5file['samples']
                    mean, std = get_inception_score(sess, images, pred_op,
                                                args.batch_size, args.splits)
                print ('mean: {} std:{}'.format(mean, std))

                res_path = h5_file.parent / 'results.csv'
                mode = 'a' if res_path.is_file() else 'w'

                with open(res_path, mode) as f:
                    if mode == 'w':
                        f.write('model,IS mean,IS std\n')
                    f.write('{},{},{}\n'.format(args.gmodel, mean, std))
                h5_file.unlink()


if __name__ == '__main__':
    main(read_args())
