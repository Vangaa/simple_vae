# Simple Variation Autoencoder
Simple [Variation Autoencoder](https://arxiv.org/pdf/1312.6114v10.pdf) for ~~complex images~~ mnist digits drawing.
![Example](https://github.com/Vangaa/simple_vae/blob/master/images/output.gif)

## Usage:
Just type `./train.py <number of latent neurons> --make_imgs` to train network and store images in `images/` folder.

To make gif file type `ffmpeg -i %03d.png output.gif -vf fps=0.1` in `images/` directory.
