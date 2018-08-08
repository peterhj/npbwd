#!/usr/bin/env python3.6

import numpy as np
np.set_printoptions(linewidth=160)

import os
import struct

class RecordTapeRandomState(object):
  def __init__(self, seed=None):
    self.state = np.random.RandomState(seed)
    self.ops = []

  def dump(self, dst):
    # TODO
    #print(self.ops)
    with open(dst, "wb") as f:
      f.write(b"RNGTAPE\x00")
      for op in self.ops:
        #print(op)
        buf = []
        assert len(op[0]) == 4
        buf.append(op[0])
        if op[0] == b"Dri.":
          buf.append(struct.pack("q", op[1]))
          buf.append(struct.pack("q", op[2]))
          buf.append(struct.pack("q", len(op[3])))
          for d in op[3]:
            buf.append(struct.pack("q", d))
          for x in np.ravel(op[4]):
            buf.append(struct.pack("q", x))
        elif op[0] == b"Dsn.":
          buf.append(struct.pack("q", len(op[1])))
          for d in op[1]:
            buf.append(struct.pack("q", d))
          for x in np.ravel(op[2]):
            buf.append(struct.pack("d", x))
        else:
          raise NotImplementedError
        buf = bytes.join(b"", buf)
        #print(len(buf), type(buf))
        assert f.write(buf) == len(buf)

  def random_integers(self, low, high=None, size=None):
    result = self.state.random_integers(low, high, size)
    self.ops.append((b"Dri.", low, high, size, result))
    return result

  def standard_normal(self, size=None):
    result = self.state.standard_normal(size)
    self.ops.append((b"Dsn.", size, result))
    return result

  def normal(self, mean, std_dev, size=None):
    result = self.state.normal(mean, std_dev, size)
    self.ops.append((b"Dn..", mean, std_dev, size, result))
    return result

def load_mnist(key, root_path="../datasets/mnist"):
  if key == "train":
    images_part = "train-images-idx3-ubyte"
    labels_part = "train-labels-idx1-ubyte"
    count = 60000
  elif key == "test":
    images_part = "t10k-images-idx3-ubyte"
    labels_part = "t10k-labels-idx1-ubyte"
    count = 10000
  else:
    assert False
  images_path = os.path.join(root_path, images_part)
  labels_path = os.path.join(root_path, labels_part)
  images_data = np.memmap(images_path, mode="r", offset=16, shape=(count, 28, 28), order="C")
  labels_data = np.memmap(labels_path, mode="r", offset=8, shape=(count,), order="C")
  print("DEBUG: mnist shapes:", images_data.shape, labels_data.shape)
  return count, images_data, labels_data

def main():
  train_count, train_images_data, train_labels_data = load_mnist("train")
  test_count, test_images_data, test_labels_data = load_mnist("test")

  #batch_sz = 2
  batch_sz = 128

  display_interval = 1
  #display_interval = 10

  #seed = None
  seed = 1234

  #rng = np.random
  #rng = np.random.RandomState(seed)
  rng = RecordTapeRandomState(seed)

  #w = rng.standard_normal((10, 28 * 28)).astype(np.float32) * 0.01
  w = rng.standard_normal((28 * 28, 10)).astype(np.float32) * 0.01
  print("DEBUG: train: init: w:", np.ravel(w)[:10])
  print("DEBUG: train: init: w:", np.ravel(w)[-10:])
  w = w.T

  iter_nr = 0
  while True:
    mb_idxs = rng.random_integers(0, train_count - 1, (batch_sz,))

    x = np.reshape(np.transpose(train_images_data[mb_idxs,:], (2, 1, 0)), (28 * 28, batch_sz)).astype(np.float32) * (1.0 / 255.0)
    kv = list(enumerate(list(train_labels_data[mb_idxs])))

    y = w.dot(x)

    y_max = np.amax(y, axis=0)
    z = np.exp(y - y_max)
    z_sum = np.sum(z, axis=0)
    p = z / z_sum

    nll = -np.log(np.array([p[v, k] for (k, v) in kv]))
    nll_sum = np.sum(nll)

    y_adj = np.array(p)
    for (k, v) in kv:
      y_adj[v, k] -= 1.0

    w_adj = y_adj.dot(x.T)

    w += w_adj * -0.003

    if (iter_nr + 1) % display_interval == 0:
      print("DEBUG: train: iters: {} loss:".format(iter_nr + 1), nll_sum / float(batch_sz))
      print("DEBUG: train:   w:", np.ravel(w)[100:110])
      print("DEBUG: train:   w:", np.ravel(w)[-110:-100])
      print("DEBUG: train:   dw:", np.ravel(w_adj)[100:110])
      print("DEBUG: train:   dw:", np.ravel(w_adj)[-110:-100])

    iter_nr += 1
    if iter_nr >= 50:
      break

  if True:
    rng.dump("mnist_tape.bin")

if __name__ == "__main__":
  main()
