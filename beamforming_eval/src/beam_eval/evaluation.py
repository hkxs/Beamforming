#  Copyright 2023 Hkxs
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the “Software”), to deal in
#  the Software without restriction, including without limitation the rights to use,
#  copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
#  Software, and to permit persons to whom the Software is furnished to do so,
#  subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.

import pandas as pd


class BeamformingEvaluation:
    """
    Class to evaluate the result of a Beamforming based on the real source and estimated source

    Properties
    ----------
    SNR: float
        Source to Noise Ratio
    SIR: float
        Source to Interference Ratio
    SDR: float
        Source to Distortion Ratio
    SAR: float
        Source to Artifacts Ratio
    """

    def __init__(self, estimated_source: pd.Series, real_source: pd.Series,
                 interference: pd.DataFrame = pd.DataFrame(), noise: pd.Series = pd.Series):
        """
        Parameters
        ----------
        estimated_source: pd.Series
            Output data from the Beamforming
        real_source: pd.Series
            Real data that without noise
        interference: pd.Dataframe
            Matrix that contains the signals that were interfering with the signal of interest, each colum should be a
            signal
        """
        self._e_artif = None
        self._e_noise = None
        self._e_interf = None
        self._s_target = None
        self.max_len = len(estimated_source)
        self.noise = None
        self.input_signal = real_source[:self.max_len].to_frame()
        self.estimated = estimated_source
        self.interference = interference[:self.max_len, :]
        self.noise = noise[:self.max_len].to_frame()

    @property
    def s_target(self):
        """
        Calculate s_{target}= P_{s_{j}}\\hat{S}_{j}

        Calculate the projection of the estimated signal (output of the beamformer) into the subspace of the real
        signal

        Returns
        -------
        pd.Series
            Target signal
        """
        if not self._s_target:
            self._s_target = self._projection(self.estimated, self.input_signal)
        return self._s_target

    @property
    def e_interf(self):
        """
        Calculate e_{interf}= P_{s}\\hat{S}_{j} - {s_target}

        Calculate the projection of the real signal into the subspace of all the input signals (signal of interest and
        interferences)

        Returns
        -------
        pd.Series
            Signal that represents the projected interferences
        """
        if not self._e_interf:
            all_inputs = pd.concat([self.input_signal, self.interference], axis=1)
            self._e_interf = self._projection(self.input_signal, all_inputs) - self.s_target
        return self._e_interf

    @property
    def e_noise(self):
        """
        Calculate e_{noise} = P_{s,n}\\hat{S}_{j} - P_{s}\\hat{S}_{j} = P_{s,n}\\hat{S}_{j} - (e_interf + s_target)

        THe noise is the projection of the real signal into the noise subspace (this subspace contains all the
        input signals and noise) minus the projection of the real signal into the subspace of all the input signals
        (also known as e_interf + s_target)

        Returns
        -------
        pd.Series
            Projected Noise signal
        """
        if not self._e_noise:
            noised_inputs = pd.concat([self.input_signal, self.interference, self.noise], axis=1)
            noise_projection = self._projection(self.input_signal, noised_inputs)
            self._e_noise = noise_projection - (self.e_interf + self.s_target)
        return self._e_noise

    @property
    def e_artif(self):
        """
        Calculate e_{artif}=\\hat{S}_{j} - P_{s,n}\\hat{S}_{j} = \\hat{S}_{j} - (e_noise + e_interf + s_target)

        The artifacts are basically the estimated signal minus all the other projections

        Returns
        -------
        pd.Series
            All the artifacts projections
        """
        if not self._e_artif:
            self._e_artif = self.estimated - (self.e_noise + self.e_interf + self.s_target)
        return self._e_artif

    @staticmethod
    def _projection(signal: pd.Series, subspace: pd.DataFrame):
        """
        Calculate the orthogonal projection of the signal into the subspace Y

        Note
        ----
        At this moment it will work only with real signals

        Parameters
        ----------
        signal: pd.Series
            Signal that will be projected into 'subspace'
        subspace: pd.Series
            Subspace where 'signal' will be projected

        Returns
        -------
        """
        return 0