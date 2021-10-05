# Data Overview

https://www.kaggle.com/c/g2net-gravitational-wave-detection/data

- Contains **simulated GW measurements** from 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo)
- The task is to identify when a signal is present in the data (target=1).
- The parameters that determine the exact form of a binary black hole waveform are the masses, sky location, distance, black hole spins, binary orientation angle, gravitational wave polarisation, time of arrival, and phase at coalescence (merger)

  - These parameters (15 in total) **have been randomised according to astrophysically motivated prior distributions and used to generate the simulated signals present in the data**, but are not provided as part of the competition data.

- Each data sample (npy file) contains 3 time series (1 for each detector) and each spans 2 sec and is sampled at 2,048 Hz.
- **The integrated signal-to noise ratio (SNR)** is classically the most informative measure of how detectable a signal is and a typical level of detectability is when this integrated SNR exceeds ~8.
  - This shouldn't confused with the instantaneous SNR - the factor by which the signal rises above the noise - and **in nearly all cases the (unlike the first gravitational wave detection GW150914) these signals are not visible by eye in the time series.**
