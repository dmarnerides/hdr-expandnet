# ExpandNet

Create HDR images from LDR inputs:

<center>

<img src="assets/examples.jpg">

| (left) LDR input | (middle) Exposures taken from the predicted HDR | (right) Tone mapped HDR |
</center>

---

Training and inference code for:

*ExpandNet: A Deep Convolutional Neural Network for High Dynamic Range Expansion from Low Dynamic Range Content*

[Demetris Marnerides](https://github.com/dmarnerides),
[Thomas Bashford-Rogers](http://thomasbashfordrogers.com/),
[Jonathan Hatchett](http://hatchett.me.uk/) 
and [Kurt Debattista](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=518)

Paper was presented at Eurographics 2018 and published in Computer Graphics Forum.

([arxiv version](https://arxiv.org/abs/1803.02266))

---

## Prerequisites

Requires the PyTorch library along with OpenCV. Python >= 3.6 supported only.

First follow the [instructions for installing PyTorch](http://pytorch.org/).

To install OpenCV use: 
```bash
conda install opencv3 -c menpo
```

---

## Usage

The expand.py script accepts an LDR image as input, and creates an HDR Image
prediction, (.hdr or .exr file if --use_exr flag is used).

```bash
python expand.py  ldr_input.jpg
```

Can also do batch processing:

```bash
python expand.py  *.jpg
```

Or take whole directory as input:

```bash
python expand.py  path/to/ldr_dir
```

To put results in a separate directory:

```bash
python expand.py  *.jpg --out results/
```

Results can be resized and filenames can be tagged:
```bash
python expand.py test.jpg --resize True --height 960 --width 540 --tag my-tag
```

If a GPU is available and Pytorch detects it then the conversion will happen on
the GPU. To force CPU usage (slower) set `--use_gpu False`.

The network operates on patches to limit memory usage when expanding high
resolution images. The flag `--patch_size` adjusts the size.

The resulting HDR Image can be tone mapped using the TMOs provided by OpenCV,
or using the exposure operator.

```bash
python expand.py  test.jpg --tone_map reinhard
```

---

## Testing on video

Use `--video True` (along with a tone mapper `--tone_map`) to test video
conversion.

**WARNING** This is completely new and experimental. Very slow and memory
consuming! Should only test short low-res clips. The model is not designed for
videos thus flickering occurs which is reduced in the code by smoothing
luminance percentile curves.

In the following example, on the left is the LDR video input and on the right
is the (reinhard) tone mapped prediction.
<center>
<img src="assets/Brzansko-Moraviste-Pejzazi_20170226_4457.gif">
</center>

Any tips to improve this are welcome!

---

## Training

To train:

```bash
python train.py 
```

For more training configuration and settings please use:

```bash
python train.py --help
```
---

## Viewing HDR Images

There is a very easy to use online viewer at [openhdr.org](https://viewer.openhdr.org/) which also has tone mapping functionality.

---

Contact: dmarnerides@gmail.com
