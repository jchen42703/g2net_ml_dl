### nnAudio Research Paper 
https://arxiv.org/pdf/1912.12055.pdf


## Q Transform
https://kinwaicheuk.github.io/nnAudio/v0.1.5/_autosummary/nnAudio.Spectrogram.CQT1992v2.html

Function is to calculate the CQT of the input signal 

Shape of Input possibilities
- (length of audio)
- (num_audio,len_audio)
- (num_audio,1,len_audio)

Returns $\rightarrow$ tensor of spectograms
- shape = 
  - (num_samples, freq_bins,time_steps) IF output_format='Magnitude'
  - (num_samples, freq_bins,time_steps, 2) IF output_format='Complex' or 'Phase'

