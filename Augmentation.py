from scipy import signal

import os
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
from scipy.interpolate import Rbf
from sklearn.utils import check_random_state
import torch
import math
from torch.nn.functional import pad, one_hot
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import random_split, Dataset, DataLoader
from random import choice

Present = True
if Present == True:  #demo function
    data = torch.load(os.path.join(os.getcwd(), 'data', 'data_Dinh_fs200.pt'))
    labels = torch.load(os.path.join(os.getcwd(), 'data', 'labels_Dinh_fs200.pt'))
    print(f"Data shape: {data.shape}")

    nb_channels = data.shape[1]
    print(f"Nb channels in a recording: {nb_channels}")

    fs = 200
    rec_time = data.shape[-1] / fs
    print(f"Recording time: {rec_time} sec")

    max_f = fs / 2
    print(f"Maximum frequency to be detected: {max_f} Hz")

    subject = 1
    channel = 22
    # x_t = data[subject][channel]
    crop_par = 10000

    x_t = data[1:5, 1:5, 0:crop_par]  # just crop the last dim to 550
    x_t_raw = data[1:5, 1:5, :]  # data without crop

    print(x_t.shape, "x_t")
    N = x_t.shape[-1]
    print(N, "N")
    t = torch.arange(0, N) * 1 / fs

    f = torch.arange(-N / 2, N / 2) * 1 / N * fs  # symmetric + normalized + appyling correct fs


def pp(x_axis, res, original_datas, name): # drawing picture for demo
    # res is list of results'list
    # list is

    for i in range(res[0].shape[0]):
        for ii in range(res[0].shape[1]):
            fig, axs = plt.subplots(5, 2)
            for iii in range(len(res)):
                num = iii
                if ii >= res[iii].shape[1]: ii = ii - 1
                re = res[iii][i][ii]
                print("i", i, "ii", ii, "iii", iii)


                if iii==0:
                   original_data = original_datas[i][ii]
                   axs[math.ceil((num + 1) / 2) - 1, num % 2].plot(x_axis, original_data, color='green', linewidth=1,
                                                                   alpha=1, label="denoised data")
                   axs[math.ceil((num + 1) / 2) - 1, num % 2].set_title("Signal_after_denoising", fontsize=20)
                   axs[math.ceil((num + 1) / 2) - 1, num % 2].legend()
                
                else:
                    original_data = original_datas[i][ii]
                    print(re.shape, name[num])
                    axs[math.ceil((num + 1) / 2) - 1, num % 2].plot(x_axis, re, color='dodgerblue', linewidth=1,
                                                                    label="transformed data")
                    #axs[math.ceil((num + 1) / 2) - 1, num % 2].plot(x_axis, original_data, color='green', linewidth=1,
                    #                                                alpha=1, label="orginal data")
                    #axs[math.ceil((num + 1) / 2) - 1, num % 2].set_title(name[num], fontsize=20)
                    #axs[math.ceil((num + 1) / 2) - 1, num % 2].legend()
                """
                original_data = original_datas[i][ii]
                print(re.shape, name[num])
                axs[math.ceil((num + 1) / 2) - 1, num % 2].plot(x_axis, re, color='orange', linewidth=0.5,
                                                                label="transformed data")
                axs[math.ceil((num + 1) / 2) - 1, num % 2].plot(x_axis, original_data, color='green', linewidth=0.5,
                                                                alpha=1, label="orginal data")
                axs[math.ceil((num + 1) / 2) - 1, num % 2].set_title(name[num], fontsize=5)
                axs[math.ceil((num + 1) / 2) - 1, num % 2].legend()
                """

            for ax in axs.flat:
                ax.set(xlabel='', ylabel='')

            # for ax in axs.flat:
            #    ax.label_outer()

            fig.savefig(str(i) + str(ii) + 'temp.png', dpi=fig.dpi)

            plt.show()


def denoise(data, s=0):
    # s=1 singel channel
    # s=0 multi channels

    # for i in range(data.shape[0] - 1):
    if s == 0:

        for i in range(data.shape[0]):
            for ii in range(data.shape[1]):
                subject = i
                channel = ii

                x_t = data[subject][channel]

                Y = fft.fftshift(fft.fft(x_t))

                N = len(x_t)
                f = torch.arange(-N / 2, N / 2) * 1 / N * fs  # symmetric + normalized + appyling correct fs
                Y_band_stop = Y.clone()

                center = int(0.5 * N / fs)
                offset = int(50 * (N / fs))
                band = int(150 * (N / fs))

                Y_band_stop[center - offset - band:center - offset + band] = 0
                Y_band_stop[center + offset - band:center + offset + band] = 0
                x_t_denoised = fft.ifft(fft.ifftshift(Y_band_stop))
                x_t_denoised = torch.real(x_t_denoised)

                data[i][ii] = x_t_denoised

        return data
    else:
        N = len(data)
        Y = torch.fft.fftshift(fft.fft(data))
        Y_band_stop = Y.clone()

        center = int(0.5 * N)
        offset = int(50 * (N / fs))
        band = int(40 * (N / fs))

        Y_band_stop[center - offset - band:center - offset + band] = 0
        Y_band_stop[center + offset - band:center + offset + band] = 0
        x_t_denoised = fft.ifft(fft.ifftshift(Y_band_stop))
        x_t_denoised = torch.real(x_t_denoised)
        data = x_t_denoised
    return data


def den(x_t):  # notch
    samp_freq = 200  # Sample frequency (Hz)
    notch_freq = 50.0  # Frequency to be removed from signal (Hz)
    quality_factor = 0.2  # Quality factor

    # Design a notch filter using signal.iirnotch
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

    # Compute magnitude response of the designed filter
    freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)

    n = np.linspace(0, 200, 55000)  # Generate 1000 sample sequence in 1 sec
    # Generate the signal containing f1 and f2
    noisySignal = x_t

    # Apply notch filter to the noisy signal using signal.filtfilt
    outputSignal = signal.filtfilt(b_notch, a_notch, noisySignal)
    return outputSignal


if Present == True:
    x_t_denoised = torch.from_numpy(den(x_t).copy())
    x_t_raw = torch.from_numpy(den(x_t_raw).copy())
    print(x_t_denoised.dtype)

    print(x_t_denoised.shape, "x_t denoised")


class Augment():
    def __init__(self, data, preaug=None):
        self.X = data  # crop data to avoid overflow
        self.preaug = preaug  # name of previous augmentation
        self.crop_par = data.shape[2]

    def _new_random_fft_phase_odd(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
        ).to(device)

        return torch.cat([
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            -torch.flip(random_phase, [-1])
        ], dim=-1)

    def _new_random_fft_phase_even(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)

        #print(rng.random((batch_size, c, n // 2 - 1)).shape)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
        ).to(device)

        return torch.cat([
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            torch.zeros((batch_size, c, 1), device=device),
            -torch.flip(random_phase, [-1])
        ], dim=-1)

    def fourier_surrogate(self,
                          channel_indep=True,
                          random_state=None
                          ):
        phase_noise_magnitude = torch.distributions.Uniform(0, 1).sample()
        if len(self.X.shape) == 3:
            batch_size = self.X.shape[0]
        else:
            batch_size = 1

        assert isinstance(
            phase_noise_magnitude,
            (Real, torch.FloatTensor, torch.cuda.FloatTensor)
        ) and 0 <= phase_noise_magnitude <= 1, (
            f"eps must be a float beween 0 and 1. Got {phase_noise_magnitude}.")

        f = torch.fft.fft(self.X.double(), dim=-1)
        device = self.X.device

        n = f.shape[-1]
        if n % 2 == 0:
            random_phase = self._new_random_fft_phase_even(
                batch_size,
                f.shape[-2] if channel_indep else 1,
                n,
                device=device,
                random_state=random_state
            )
        else:
            random_phase = self._new_random_fft_phase_odd(
                batch_size,
                f.shape[-2] if channel_indep else 1,
                n,
                device=device,
                random_state=random_state
            )
        if not channel_indep:
            random_phase = torch.tile(random_phase, (1, f.shape[-2], 1))
        if isinstance(phase_noise_magnitude, torch.Tensor):
            phase_noise_magnitude = phase_noise_magnitude.to(device)
        f_shifted = f * torch.exp(phase_noise_magnitude * random_phase)
        shifted = torch.fft.ifft(f_shifted, dim=-1)
        transformed_X = shifted.real.float()

        return transformed_X

    def _analytic_transform(self, x):
        if torch.is_complex(x):
            raise ValueError("x must be real.")

        N = x.shape[-1]
        f = torch.fft.fft(x, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2

        return torch.fft.ifft(f * h, dim=-1)

    def _nextpow2(self, n):

        return int(np.ceil(np.log2(np.abs(n))))

    def _frequency_shift(self, X, fs, f_shift):

        # Pad the signal with zeros to prevent the FFT invoked by the transform
        # from slowing down the computation:
        n_channels, N_orig = X.shape[-2:]
        N_padded = 2 ** self._nextpow2(N_orig)
        t = torch.arange(N_padded, device=X.device) / fs
        padded = pad(X, (0, N_padded - N_orig))
        analytical = self._analytic_transform(padded)
        if isinstance(f_shift, (float, int, np.ndarray, list)):
            f_shift = torch.as_tensor(f_shift).float()
        reshaped_f_shift = f_shift.repeat(
            N_padded, n_channels, 1).T
        shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
        return shifted[..., :N_orig].real.float()

    def frequency_shift(self, delta_freq=0.004, sfreq=15.0):

        transformed_X = self._frequency_shift(
            X=self.X,
            fs=sfreq,
            f_shift=delta_freq,
        )
        return transformed_X

    def gaussian_jitter(self, std=0.00001, random_state=None):
        X = self.X
        rng = check_random_state(random_state)
        if isinstance(std, torch.Tensor):
            std = std.to(X.device)
        noise = torch.from_numpy(
            rng.normal(
                loc=np.zeros(X.shape),
                scale=1
            ),
        ).float().to(X.device) * std
        transformed_X = X + noise
        return transformed_X

    def sign_flip(self):
        data = self.X

        return -data

    def time_flip(self):
        data = self.X
        return torch.flip(data, [-1])

    def smooth_time_mask(self):
        X = self.X

        mask_start_per_sample = torch.tensor(100)
        mask_len_samples = X.shape[-1] / 5
        """Smoothly replace a contiguous part of all channels by zeros.
        Originally proposed in [1]_ and [2]_
        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor
            EEG labels for the example or batch.
        mask_start_per_sample : torch.tensor
            Tensor of integers containing the position (in last dimension) where to
            start masking the signal. Should have the same size as the first
            dimension of X (i.e. one start position per example in the batch).
        mask_len_samples : int
            Number of consecutive samples to zero out.
        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor
            Transformed labels.
        References
        ----------
        .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
           Subject-aware contrastive learning for biosignals. arXiv preprint
           arXiv:2007.04871.
        .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
           Representation Learning for Electroencephalogram Classification. In
           Machine Learning for Health (pp. 238-253). PMLR.
        """
        batch_size = X.shape[0]
        n_channels = X.shape[1]
        seq_len = X.shape[2]
        t = torch.arange(seq_len, device=X.device).float()
        t = t.repeat(batch_size, n_channels, 1)
        mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
        s = 1000 / seq_len
        mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
                torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))
                ).float().to(X.device)

        return X * mask

    def channels_permute(self):
        permutation = [1, 1, 1]
        X = self.X
        """Permute EEG channels according to fixed permutation matrix.
        Suggested e.g. in [1]_
        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor
            EEG labels for the example or batch.
        permutation : list
            List of integers defining the new channels order.
        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor
            Transformed labels.
        References
        ----------
        .. [1] Deiss, O., Biswal, S., Jin, J., Sun, H., Westover, M. B., & Sun, J.
           (2018). HAMLET: interpretable human and machine co-learning technique.
           arXiv preprint arXiv:1803.09702.
        """
        return X[..., permutation, :]

    def _pick_channels_randomly(self, X, p_pick, random_state):
        rng = check_random_state(random_state)
        batch_size, n_channels, _ = X.shape
        # allows to use the same RNG
        unif_samples = torch.as_tensor(
            rng.uniform(0, 1, size=(batch_size, n_channels)),
            dtype=torch.float,
            device=X.device,
        )
        # equivalent to a 0s and 1s mask
        return torch.sigmoid(1000 * (unif_samples - p_pick))

    def _make_permutation_matrix(self, X, mask, random_state):
        rng = check_random_state(random_state)
        batch_size, n_channels, _ = X.shape
        hard_mask = mask.round()
        batch_permutations = torch.empty(
            batch_size, n_channels, n_channels, device=X.device
        )
        for b, mask in enumerate(hard_mask):
            channels_to_shuffle = torch.arange(n_channels)
            channels_to_shuffle = channels_to_shuffle[mask.bool()]
            channels_permutation = np.arange(n_channels)
            channels_permutation[channels_to_shuffle] = rng.permutation(
                channels_to_shuffle
            )
            channels_permutation = torch.as_tensor(
                channels_permutation, dtype=torch.int64, device=X.device
            )
            batch_permutations[b, ...] = one_hot(channels_permutation)
        return batch_permutations

    def channels_shuffle(self, random_state=None):
        X = self.X
        a = np.random.random()

        p_shuffle = a

        """Randomly shuffle channels in EEG data matrix.
        Part of the CMSAugment policy proposed in [1]_
        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor
            EEG labels for the example or batch.
        p_shuffle: float | None
            Float between 0 and 1 setting the probability of including the channel
            in the set of permutted channels.
        random_state: int | numpy.random.Generator, optional
            Seed to be used to instantiate numpy random number generator instance.
            Used to sample which channels to shuffle and to carry the shuffle.
            Defaults to None.
        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor
            Transformed labels.
        References
        ----------
        .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
           Learning from Heterogeneous EEG Signals with Differentiable Channel
           Reordering. arXiv preprint arXiv:2010.13694.
        """
        if p_shuffle == 0:
            return X
        mask = self._pick_channels_randomly(X, 1 - p_shuffle, random_state)
        batch_permutations = self._make_permutation_matrix(X, mask, random_state)
        # print(type(batch_permutations),batch_permutations.shape)
        return torch.matmul(batch_permutations.float(), X.float())

    def scaling(self, random_state=None):
        X = self.X
        p_shuffle = 0.6
        return torch.mul(p_shuffle, X.float())

    def window_slicing(self, random_state=None):
        X = self.X
        p_shuffle = 200

        return x_t_raw[..., p_shuffle:self.crop_par + p_shuffle]

    def window_wrap(self):
        '''
        xx=self.X
        x_raw=x_t_raw
        gap_s=0
        v_square=1
        for i in range(xx.shape[-1]):
            if i ==4:
                print()
            gap_s=v_square*i
            gap = i + int(gap_s)
            if gap_s+i<x_t_raw.shape[-1]:

                a=xx[...,i]
                b=x_raw[...,gap]
                xx[...,i]=x_raw[...,gap]
            else:
                print("exceed")
        return xx
        '''
        X = self.X
        p_shuffle = 600

        return x_t_raw[..., p_shuffle:self.crop_par + p_shuffle]

    def sselct(self):
        # this function will random use one augmentation algorithm below
        ll = [self.fourier_surrogate, self.frequency_shift, self.gaussian_jitter, self.sign_flip, self.time_flip,
              self.channels_permute, self.scaling,self.window_wrap,self.window_slicing,self.smooth_time_mask]

        # ll=[self.fourier_surrogate, self.frequency_shift]

        if Present == True:
            name_func = []
            res = []
            for l in range(len(ll)):
                name_func.append(ll[l].__name__)
                res.append(ll[l]())
            return res, name_func
        else:
            if self.preaug != None:
                # print("no", ll[self.preaug].__name__,"next time")  # delete the preaugmentation from list
                ll.pop(self.preaug)
            res = choice(range(len(ll)))
            return ll[res](), res, ll



if Present == True:
    trans_test_res, trans_name = Augment(x_t_denoised).sselct()
    N = trans_test_res[0].shape[-1]
    t = torch.arange(0, N) * 1 / fs
    pp(t, trans_test_res, x_t_denoised, trans_name)


