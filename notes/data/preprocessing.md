# Data Preprocessing

Large doc for notes on preprocessing.

- Should primarily be on methods used in the kaggle challenge

# Data Preprocessing with GWPy

From: https://www.kaggle.com/mistag/data-preprocessing-with-gwpy

## Procedures

- Apply a window function (Tukey - tapered cosine window) to suppress spectral leakage
- Whiten the spectrum
- Bandpass
  - A band-pass filter or bandpass filter is a device that passes frequencies within a certain range and rejects frequencies outside that range.
- Q-Transform
  - a Constant Q transform will yield better results (than a Discrete Fourier Transform) where low frequencies and logarithmic frequency mapping are concerned, but it's extremely difficult to make it run realtime and has slightly less detail in the far upper frequencies.
- Combine three channels into one RGB image
- Used as input for CNN

# 1st Place Solution

- DL Part: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275476
- Preprocessing Part: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275507

## Preprocessing

- Butterworth filter
- double application (filtfilt) of 5th order high-pass filter with critical frequency at 20.43Hz

  - can freely decrease order down to 2 and critical frequency down to 15Hz as soon as NN training got some traction, but no real performance benefits

- any additional filtering PSD of data was obtained using a modified Welch method
- skipped whitening
- In the 1D approach NN should derive filters from data and we had no tangible way to "preload" it with a calculated Matched Filter bank, so why bother.
  - whitening is equivalent to applying an arbitrary frequency-domain filter and would mess up portions of a signal.
- In general, pipeline first and last few seconds of a signal was just discarded, but few seconds are all we have here.
- Later in competition we found a way to “expand” a signal enough to apply whitening, but at that point we felt no need to use it, as separate encoders kicked in.

## Make Some Noise

## Improved Conv1D model (0.883 public LB)

...

## Using synthetic dataset (0.886 public LB)

...

# 4th Place Solution

https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275331

## 1D Model

- simple Conv1d architecture with just 8 conv1d layers along with a normal 2x linear head , to our surprise it scored really well cv 0.8766 Lb 0.8788 .
- train deep conv1d model with residuals (similar to resnet) and it worked like a charm , we then changed the head from linear to LSTM and got further boost
- GW waves numpy array --> horizontal stacking all three to get (1,4096\*3) array --> band pass filtering ---> Deep conv1d backbone with residuals --->LSTM head ---> Prediction

## 2D Model

```
qtransform_params={"sr": 2048, "fmin": 30, "fmax": 400, "hop_length": 4,
"bins_per_octave": 12, "filter_scale" : 0.3}
```

- Sequence of Preprocessing is as follows:

```
  Numpy ---> signal tukey ---> band pass filter ---> normalized by norm_by =[7.729773e-21,8.228142e-21, 8.750003e-21] --->CQT--> Augmentations[coloredNoise and shift]
```

- Torch audiomentations were used to apply augmentations. Colored noise augmentation was done channel wise while shift was applied sample wise.

# Notebooks To Look Into

- Reverse engineering: Create clean GW signals
- https://www.kaggle.com/allunia/signal-where-are-you
- https://www.kaggle.com/headsortails/when-stars-collide-g2net-eda
- https://www.kaggle.com/yasufuminakama/g2net-efficientnet-b7-baseline-training
- Learnable CQT
  - https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275318

# Notable Resources

- [GWPy: Library for studying GW data](https://gwpy.github.io/docs/latest/index.html)
- Tutorials
  - https://www.gw-openscience.org/tutorials/
  - https://www.gw-openscience.org/software/
  - https://iopscience.iop.org/article/10.1088/1361-6382/ab685e
- https://github.com/analokmaus/kaggle-g2net-public
