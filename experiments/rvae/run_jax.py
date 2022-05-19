import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.nn.initializers import glorot_normal, normal, ones
import craystack as cs
import itertools
from functools import partial
import numpy as onp
from autograd.builtins import tuple as ag_tuple
from model_jax import rvae_codec
import utils
from datasets_jax import CIFAR10
from tqdm import tqdm


prior_precision = 10
x_precision = 24
q_precision = 18
l1 = 1000000


# build input pipeline
total_bs = 64
device_bs = total_bs // jax.device_count()
dataset = CIFAR10(class_conditional=False, randflip=True)  # note augmentations not used for eval
eval_ds = dataset.get_eval_dataset(
    batch_shape=(
        jax.local_device_count(),  # for pmap
        device_bs,  # batch size per device
    ))
eval_iter = utils.numpy_iter(eval_ds)


x_shape = onp.array([device_bs, 32, 32, 3])  # expect image to be NHWC
latent_shape = onp.array([x_shape[0], x_shape[1]//2, x_shape[2]//2, 32])
x_size = jnp.prod(x_shape)
latent_size = jnp.prod(latent_shape)

def vae_view(head):
  return ag_tuple((jnp.reshape(head[:latent_size], latent_shape),
                   jnp.reshape(head[latent_size:], x_shape)))

codec = cs.substack(rvae_codec(x_precision, prior_precision,
                               q_precision, x_shape),
                    vae_view)

rng = onp.random.RandomState(0)
initial_message = cs.random_message(l1, (x_size + latent_size,), rng)
message = initial_message

for ebatch in tqdm(eval_iter):
    # message = jax.pmap(compress_fn)(ebatch['image'])  # (16, 32, 32, 3)
    image = ebatch[0]
    message = codec.push(message, image)
    break
l2 = len(cs.flatten(message))
print('{} bits per dim'.format(32 * (l2 - l1) / x_size))
message_, image_ = codec.pop(message)
onp.testing.assert_equal(image, image_)
onp.testing.assert_equal(cs.flatten(initial_message), cs.flatten(message_))
