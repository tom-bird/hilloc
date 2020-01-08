This repository accompanies the paper [HiLLoC: Lossless Image Compression with Hierarchical Latent Variable Models](https://openreview.net/forum?id=r1lZgyBYwS) by [James Townsend](https://j-towns.github.io), [Thomas Bird](https://tom-bird.github.io/), [Julius Kunze](https://juliuskunze.com/) and [David Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php), appearing at ICLR '20.

The Craystack software for lossless compression can be found at https://github.com/j-towns/craystack, code to reproduce the experiments in the paper is in the [experiments](experiments) directory.

We recommend using the [colab notebook](https://colab.research.google.com/drive/11967hjFQczjW21cLLTFhOnTurx3mSBVD), which demonstrates compression using HiLLoC and a ResNet VAE model. The notebook is a minimal implementation written in [JAX](https://github.com/google/jax), taking the trained weights resulting from the tensorflow implementation in the experiments directory. You can download trained weights for the [4 layer](https://drive.google.com/open?id=1PXZAGdA-PswnxcJNla6WBIGAnuRcdJRY) and [24 layer](https://drive.google.com/open?id=15EtLr19cw5yxg4B26dqRe-3-kNyh-z4O) model, as well as a [sample cifar image](https://drive.google.com/open?id=11w3rJXm111zXMlZ8d3GwNpI_-9U24vRw). Feel free to compress your own images!
