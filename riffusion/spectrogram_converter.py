import warnings

import numpy as np
import pydub
import torch
import torchaudio

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import audio_util, torch_util


class SpectrogramConverter:
    """
    Convert between audio segments and spectrogram tensors using torchaudio.

    In this class a "spectrogram" is defined as a (batch, time, frequency) tensor with float values
    that represent the amplitude of the frequency at that time bucket (in the frequency domain).
    Frequencies are given in the perceptul Mel scale defined by the params. A more specific term
    used in some functions is "mel amplitudes".

    The spectrogram computed from `spectrogram_from_audio` is complex valued, but it only
    returns the amplitude, because the phase is chaotic and hard to learn. The function
    `audio_from_spectrogram` is an approximate inverse of `spectrogram_from_audio`, which
    approximates the phase information using the Griffin-Lim algorithm.

    Each channel in the audio is treated independently, and the spectrogram has a batch dimension
    equal to the number of channels in the input audio segment.

    Both the Griffin Lim algorithm and the Mel scaling process are lossy.

    For more information, see https://pytorch.org/audio/stable/transforms.html
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params

        self.device = torch_util.check_device(device)

        if device.lower().startswith("mps"):
            warnings.warn(
                "WARNING: MPS does not support audio operations, falling back to CPU for them",
                stacklevel=2,
            )
            self.device = "cpu"

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            n_stft=params.n_fft // 2 + 1,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            # max_iter=params.max_mel_iters,
            # tolerance_loss=1e-5,
            # tolerance_change=1e-8,
            # sgdargs=None,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

    def spectrogram_from_audio(
        self,
        audio: pydub.AudioSegment,
    ) -> np.ndarray:
        """
        Compute a spectrogram from an audio segment.

        Args:
            audio: Audio segment which must match the sample rate of the params

        Returns:
            spectrogram: (channel, frequency, time)
        """
        assert int(audio.frame_rate) == self.p.sample_rate, "Audio sample rate must match params"

        # Get the samples as a numpy array in (batch, samples) shape
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])

        # Convert to floats if necessary
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_tensor = torch.from_numpy(waveform).to(self.device)
        amplitudes_mel = self.mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()

    def audio_from_spectrogram(
        self,
        spectrogram: np.ndarray,
        apply_filters: bool = True,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram.

        Args:
            spectrogram: (batch, frequency, time)
            apply_filters: Post-process with normalization and compression

        Returns:
            audio: Audio segment with channels equal to the batch dimension
        """
        # Move to device
        amplitudes_mel = torch.from_numpy(spectrogram).to(self.device)

        # Reconstruct the waveform
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)

        # Convert to audio segment
        segment = audio_util.audio_from_waveform(
            samples=waveform.cpu().numpy(),
            sample_rate=self.p.sample_rate,
            # Normalize the waveform to the range [-1, 1]
            normalize=True,
        )

        # Optionally apply post-processing filters
        if apply_filters:
            segment = audio_util.apply_filters(
                segment,
                compression=False,
            )

        return segment

    def mel_amplitudes_from_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to compute Mel-scale amplitudes from a waveform.

        Args:
            waveform: (batch, samples)

        Returns:
            amplitudes_mel: (batch, frequency, time)
        """
        # Compute the complex-valued spectrogram
        spectrogram_complex = self.spectrogram_func(waveform)

        # Take the magnitude
        amplitudes = torch.abs(spectrogram_complex)

        # Convert to mel scale
        return self.mel_scaler(amplitudes)

    def waveform_from_mel_amplitudes(
        self,
        amplitudes_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to approximately reconstruct a waveform from Mel-scale amplitudes.

        Args:
            amplitudes_mel: (batch, frequency, time)

        Returns:
            waveform: (batch, samples)
        """
        # Convert from mel scale to linear
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)

        # Run the approximate algorithm to compute the phase and recover the waveform
        return self.inverse_spectrogram_func(amplitudes_linear)
# import numpy as np
# from moviepy.editor import AudioFileClip
# from moviepy.audio.AudioClip import AudioArrayClip
# # from moviepy.audio.fx.all import audio_fadein, audio_fadeout
# import tempfile
# class SpectrogramConverter:
#     def __init__(self, params, device="cuda"):
#         self.p = params
#         self.device = device  # We are not using this, but keeping for compatibility
#
#     def spectrogram_from_audio(self, audio_path):
#         """
#         Compute a spectrogram from an audio segment.
#         Args:
#             audio_path: Path to the audio file
#         Returns:
#             spectrogram: (frequency, time)
#         """
#         audio_clip = AudioFileClip(audio_path)
#         samples = audio_clip.to_soundarray(fps=self.p.sample_rate)
#
#         return self.mel_amplitudes_from_waveform(samples)
#
#     def audio_from_spectrogram(self, spectrogram, duration, apply_filters=True):
#         """
#         Reconstruct an audio segment from a spectrogram.
#         Args:
#             spectrogram: (frequency, time)
#             duration: Duration of the audio in seconds
#             apply_filters: Post-process with normalization and compression
#         Returns:
#             audio: Audio segment as a numpy array
#         """
#         waveform = self.waveform_from_mel_amplitudes(spectrogram)
#         audio_clip = AudioArrayClip(waveform, fps=self.p.sample_rate).set_duration(duration)
#         if apply_filters:
#             audio_clip = self.apply_filters(audio_clip)
#         return audio_clip
#
#     def mel_amplitudes_from_waveform(self, waveform):
#         """
#         Convert waveform to mel-scale amplitudes.
#         Args:
#             waveform: (samples, channels)
#         Returns:
#             amplitudes_mel: (frequency, time)
#         """
#         # This is a placeholder. Implement actual mel conversion if needed.
#         return np.abs(np.fft.rfft(waveform, axis=0))
#
#     def waveform_from_mel_amplitudes(self, amplitudes_mel):
#         """
#         Convert mel-scale amplitudes back to waveform.
#         Args:
#             amplitudes_mel: (frequency, time)
#         Returns:
#             waveform: (samples, channels)
#         """
#         # This is a placeholder. Implement actual inverse mel conversion if needed.
#         return np.fft.irfft(amplitudes_mel, axis=0)

    # def apply_filters(self, audio_clip):
    #     """
    #     Apply post-processing filters to the audio segment.
    #     Args:
    #         audio_clip: MoviePy AudioClip
    #     Returns:
    #         Filtered audio clip
    #     """
    #     audio_clip = audio_fadein(audio_clip, 1)  # Fade in over 1 second
    #     audio_clip = audio_fadeout(audio_clip, 1)  # Fade out over 1 second
    #     return audio_clip