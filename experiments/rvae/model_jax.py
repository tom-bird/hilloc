from jax import lax, random, jit, nn
import numpy as onp
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.stats import logistic
from jax.nn.initializers import glorot_normal, normal, ones
import craystack as cs
import itertools
from functools import partial
from autograd.builtins import tuple as ag_tuple


z_size = 32
h_size = 160
depth = 24

params = onp.load('rvae_weights_{}layer.npy'.format(depth), allow_pickle=True).item()
nparams = sum([p.size for p in params.values()])
print('{} parameters'.format(nparams))

dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
one = (1, 1)

conv = partial(lax.conv_general_dilated,
               dimension_numbers=dimension_numbers,
               padding='SAME')


def l2_normalize(x, axis, epsilon=1e-12):
    l2 = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    l2 = jnp.sqrt(jnp.maximum(l2, epsilon))
    return x / l2


def apply_conv(params, inputs, strides=one):
    W, g, b = params
    W = jnp.exp(g) * l2_normalize(W, axis=[0, 1, 2])  # weight norm
    return conv(inputs, W, strides) + b


def empty_list(l):
    return [None for i in range(l)]


def get_params(name, layer=None):
    if layer is not None:
        s = 'model/IAF_0_{}/{}/'.format(layer, name)
    else:
        s = 'model/{}/'.format(name)
    W = params[s + 'V:0']
    g = params[s + 'g:0']
    b = params[s + 'b:0']
    return W, g, b


## ------------------------- MODEL ---------------------------------

def down_split(layer, inputs):
    out = nn.elu(inputs)
    out = apply_conv(get_params('down_conv1', layer), out)
    prior_mean, prior_logstd, rz_mean, rz_logstd, _, h_det = \
        jnp.split(out, [z_size, 2 * z_size, 3 * z_size, 4 * z_size,
                       4 * z_size + h_size], axis=-1)
    return (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det


def down_merge(layer, h_det, inputs, z):
    h = jnp.concatenate([z, h_det], axis=-1)
    h = nn.elu(h)
    h = apply_conv(get_params('down_conv2', layer), h)
    inputs = inputs + 0.1 * h
    return inputs


def up_pass(x):
    inputs = jnp.clip((x.astype('float64') + 0.5) / 256.0, 0.0, 1.0) - 0.5
    inputs = apply_conv(get_params('x_enc'), inputs, strides=(2, 2))

    q_stats = empty_list(depth)  # these are all we care about from the up pass
    for i in range(depth):
        # up split
        out = nn.elu(inputs)
        out = apply_conv(get_params('up_conv1', i), out)
        qz_mean, qz_logstd, _, h = jnp.split(out, [z_size, 2 * z_size, 2 * z_size + h_size], axis=-1)
        q_stats[i] = qz_mean, qz_logstd

        # up merge
        h = nn.elu(h)
        h = apply_conv(get_params('up_conv3', i), h)
        inputs = inputs + 0.1 * h
    return q_stats


def upsample_and_postprocess(inputs):
    out = nn.elu(inputs)
    W, g, b = get_params('x_dec')
    # W is HWOI rather than HWIO, and gets normalised incorrectly it seems
    W = jnp.exp(g).reshape(1, 1, -1, 1) * l2_normalize(W, axis=(0, 1, 2))
    # transpose_kernel for compatibility with tf conv transpose
    out = lax.conv_transpose(out, W, dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                             strides=(2, 2), padding='SAME',
                             transpose_kernel=True)
    out += b
    return jnp.clip(out, -0.5 + 1 / 512., 0.5 - 1 / 512.)


def rvae_codec(x_precision, prior_prec, posterior_prec, data_shape):
    batch_size, h, w, c = data_shape
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    x_logstd = params['model/dec_log_stdv:0']
    h_init = params['model/h_top:0']

    prior_codec = cs.substack(cs.Uniform(prior_prec), z_view)

    def post_codec(post_mean, post_stdd, prior_mean, prior_stdd):
        return cs.substack(cs.DiagGaussian_GaussianBins(post_mean, post_stdd,
                                                        prior_mean, prior_stdd, posterior_prec, prior_prec), z_view)

    def x_codec(x_mean):
        return cs.substack(
            cs.Logistic_UnifBins(x_mean, x_logstd, x_precision,
                                 bin_prec=8, bin_lb=-0.5, bin_ub=0.5), x_view)

    def push(message, data):
        q_stats = up_pass(data)

        # run down pass and pop according to posterior, top down
        inputs = jnp.tile(jnp.reshape(h_init, (1, 1, 1, -1)),
                          (batch_size, h // 2, w // 2, 1))
        zs = empty_list(depth)
        for i in reversed(range(depth)):
            (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det = \
                down_split(i, inputs)
            qz_mean, qz_logstd = q_stats[i]

            codec = post_codec(qz_mean + rz_mean, jnp.exp(qz_logstd + rz_logstd),
                               prior_mean, jnp.exp(prior_logstd))
            message, z = codec.pop(message)
            zs[i] = z
            z = prior_mean + \
                cs.std_gaussian_centres(prior_prec)[z] * jnp.exp(prior_logstd)
            inputs = down_merge(i, h_det, inputs, z)

        # push data
        x_mean = upsample_and_postprocess(inputs)
        codec = x_codec(x_mean)
        message, = codec.push(message, data)

        # push z according to prior, bottom up
        for i in range(depth):
            message, = prior_codec.push(message, zs[i])
        return message,

    def pop(message):
        # pop z according to prior, top down
        zs = empty_list(depth)
        rp_stats = empty_list(depth)
        inputs = jnp.tile(jnp.reshape(h_init, (1, 1, 1, -1)),
                          (batch_size, h // 2, w // 2, 1))
        for i in reversed(range(depth)):
            message, z = prior_codec.pop(message)
            (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det = \
                down_split(i, inputs)
            zs[i] = z
            rp_stats[i] = prior_mean, prior_logstd, rz_mean, rz_logstd
            z = prior_mean + \
                cs.std_gaussian_centres(prior_prec)[z] * jnp.exp(prior_logstd)
            inputs = down_merge(i, h_det, inputs, z)

        # pop data
        x_mean = upsample_and_postprocess(inputs)
        codec = x_codec(x_mean)
        message, data = codec.pop(message)

        # push z according to posterior, bottom up
        q_stats = up_pass(data)
        for i in range(depth):
            qz_mean, qz_logstd = q_stats[i]
            prior_mean, prior_logstd, rz_mean, rz_logstd = rp_stats[i]
            codec = post_codec(qz_mean + rz_mean, jnp.exp(qz_logstd + rz_logstd),
                               prior_mean, jnp.exp(prior_logstd))
            message, = codec.push(message, zs[i])
        return message, data

    return cs.Codec(push, pop)


def observed_rvae_codec(x_precision, prior_prec, posterior_prec, data_shape):
    batch_size, h, w, c = data_shape
    latent_shape = onp.array([data_shape[0], data_shape[1] // 2, data_shape[2] // 2, 32])
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    x_logstd = params['model/dec_log_stdv:0']
    h_init = params['model/h_top:0']

    prior_codec = cs.substack(cs.Uniform(prior_prec), z_view)

    def post_codec(post_mean, post_stdd, prior_mean, prior_stdd):
        return cs.substack(cs.DiagGaussian_GaussianBins(post_mean, post_stdd,
                                                        prior_mean, prior_stdd, posterior_prec, prior_prec), z_view)

    def x_codec(x_mean):
        return cs.substack(
            cs.Logistic_UnifBins(x_mean, x_logstd, x_precision,
                                 bin_prec=8, bin_lb=-0.5, bin_ub=0.5), x_view)

    def push(message, data):
        # dummy_msg = dummy_message
        in_message = message
        q_stats = up_pass(data)

        zbits = 0.

        # run down pass and pop according to posterior, top down
        inputs = jnp.tile(jnp.reshape(h_init, (1, 1, 1, -1)),
                          (batch_size, h // 2, w // 2, 1))
        zs = empty_list(depth)
        for i in reversed(range(depth)):
            (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det = \
                down_split(i, inputs)
            qz_mean, qz_logstd = q_stats[i]

            codec = post_codec(qz_mean + rz_mean, jnp.exp(qz_logstd + rz_logstd),
                               prior_mean, jnp.exp(prior_logstd))
            message, z = codec.pop(message)
            zs[i] = z
            z = prior_mean + \
                cs.std_gaussian_centres(prior_prec)[z] * jnp.exp(prior_logstd)
            inputs = down_merge(i, h_det, inputs, z)

            zbits += - norm.logpdf(z, prior_mean, jnp.exp(prior_logstd)) / onp.log(2.)

        # reset the message
        message = in_message

        # push data
        x_mean = upsample_and_postprocess(inputs)
        codec = x_codec(x_mean)
        message, = codec.push(message, data)

        norm_x = jnp.clip((data.astype('float64') + 0.5) / 256.0, 0.0, 1.0) - 0.5
        obits = - logistic.logpdf(norm_x, x_mean, jnp.exp(x_logstd)) / onp.log(2.)

        print(onp.sum(zbits) / onp.prod(data_shape))
        print(onp.sum(obits) / onp.prod(data_shape))

        cbits = onp.sum(zbits) + onp.sum(obits)

        print('continuous bits per dim: {}'.format(cbits / onp.prod(data_shape)))

        # push z according to prior, bottom up
        for i in range(depth):
            message, = prior_codec.push(message, zs[i])
        return message,

    def pop(message):
        # pop z according to prior, top down
        zs = empty_list(depth)
        rp_stats = empty_list(depth)
        inputs = jnp.tile(jnp.reshape(h_init, (1, 1, 1, -1)),
                          (batch_size, h // 2, w // 2, 1))
        for i in reversed(range(depth)):
            message, z = prior_codec.pop(message)
            (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det = \
                down_split(i, inputs)
            zs[i] = z
            rp_stats[i] = prior_mean, prior_logstd, rz_mean, rz_logstd
            z = prior_mean + \
                cs.std_gaussian_centres(prior_prec)[z] * jnp.exp(prior_logstd)
            inputs = down_merge(i, h_det, inputs, z)

        # pop data
        x_mean = upsample_and_postprocess(inputs)
        codec = x_codec(x_mean)
        message, data = codec.pop(message)

        # push z according to posterior, bottom up
        # q_stats = up_pass(data)
        # for i in range(depth):
        #     qz_mean, qz_logstd = q_stats[i]
        #     prior_mean, prior_logstd, rz_mean, rz_logstd = rp_stats[i]
        #     codec = post_codec(qz_mean + rz_mean, jnp.exp(qz_logstd + rz_logstd),
        #                        prior_mean, jnp.exp(prior_logstd))
        #     message, = codec.push(message, zs[i])
        return message, data

    return cs.Codec(push, pop)
