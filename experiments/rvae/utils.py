import functools

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as onp
import PIL
import math


def tf_to_numpy(tf_batch):
  """TF to NumPy, using ._numpy() to avoid copy."""
  # pylint: disable=protected-access,g-long-lambda
  return jax.tree_map(lambda x: (x._numpy()
                                 if hasattr(x, '_numpy') else x), tf_batch)


def numpy_iter(tf_dataset):
  return map(tf_to_numpy, iter(tf_dataset))
