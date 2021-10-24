# Signal Processing

Goes over common terms and their definitions.

# Wavelets

https://towardsdatascience.com/what-is-wavelet-and-how-we-use-it-for-data-science-d19427699cef

## The Idea of Wavelets

- Normally, fourier transforms let you decompose a signal into a frequency spectrum
  - But, then we lose the time resolution of the original signal
  - Can use Short-Time Fourier Transforms (apply FT to signals that are divided up), but can't catch information about signals with frequencies lower than 1 Hz (1 cycle per second)
- In python (i.e. numpy), programs assume the duration of your signal is 1 second.
- Need a bigger time window to catch low frequency and smaller window for high frequency

# Wavelets (2)

https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34

## Wavelets

- Properties
  - Scale (Stretching/Squishing) of wavelet
  - Location
- **Wave is an oscillating function of time or space and is periodic. In contrast, wavelets are localized waves.**
- A wavelet is a wave-like oscillation with an amplitude that begins at zero, increases, and then decreases back to zero.

## Purpose of Wavelet Transforms

- Normally, fourier transforms let you decompose a signal into frequency spectrum that represent the entire signal
- Wavelets let us decompose a function signal into a set of wavelets

## How do the transforms work?

**Idea:** compute how much of a wavelet is in a signal for a particular scale and location

- It's a convolution.

A signal is convolved with a set of wavelets at a variety of scales.
