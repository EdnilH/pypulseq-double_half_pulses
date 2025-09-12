import numpy as np
import pypulseq as pp
from typing import Tuple, Union
from types import SimpleNamespace


def make_half_sinc_pulse(flip_angle: float,
                         side: str,
                         apodization: float = 0,
                         delay: float = 0,
                         duration: float = 4e-3,
                         dwell: float = 0,
                         center_pos: float = 0.5,
                         freq_offset: float = 0,
                         max_grad: float = 0,
                         max_slew: float = 0,
                         phase_offset: float = 0,
                         return_gz: bool = False,
                         slice_thickness: float = 0,
                         system: Union[pp.Opts, None] = None,
                         time_bw_product: float = 4,
                         use: str = str()
                         ) -> Union[
                                    SimpleNamespace,
                                    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace],
                                    ]:
    """
        Creates a radio-frequency half sinc pulse event.

        Parameters
        ----------
        flip_angle : float
           Flip angle in radians.
        side : str
            Side of the pulse to be returned (left/right)
        apodization : float, default=0
           Apodization.
        center_pos : float, default=0.5
           Position of peak 0.5 (midway).
        delay : float, default=0
           Delay in seconds (s).
        duration : float, default=4e-3
           Duration in seconds (s).
        dwell : float, default=0
        freq_offset : float, default=0
           Frequency offset in Hertz (Hz).
        max_grad : float, default=0
           Maximum gradient strength of accompanying slice select trapezoidal event.
        max_slew : float, default=0
           Maximum slew rate of accompanying slice select trapezoidal event.
        phase_offset : float, default=0
           Phase offset in Hertz (Hz).
        return_gz : bool, default=False
           Boolean flag to indicate if slice-selective gradient has to be returned.
        slice_thickness : float, default=0
           Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
           slice select event.
        system : Opts, default=Opts()
           System limits. Default is a system limits object initialized to default values.
        time_bw_product : float, default=4
           Time-bandwidth product.
        use : str, default=str()
           Use of radio-frequency sinc pulse. Must be one of 'excitation', 'refocusing' or 'inversion'.

        See also `pypulseq.Sequence.sequence.Sequence.add_block()`.
        See also `pypulseq.make_sinc_pulse module`.

        Returns
        -------
        rf : SimpleNamespace
           Radio-frequency sinc pulse event.
        gz : SimpleNamespace, optional
           Accompanying slice select trapezoidal gradient event. Returned only if `slice_thickness` is provided.
        gzr : SimpleNamespace, optional
           Accompanying slice select rephasing trapezoidal gradient event. Returned only if `slice_thickness` is provided.

        Raises
        ------
        ValueError
           If invalid `side` parameter was passed. Must be one of 'left' or 'right'
           If length of the full sinc pulse is uneven.
        NotImplementedError
            If `return_gz` is set to True
        """
    if return_gz:
        raise NotImplementedError("Creating the slice select gradient for half sinc pulses has not been implemented yet")
    valid_side_uses = ["left", "right"]
    if side not in valid_side_uses:
        raise ValueError(f"Invalid side parameter. Must be one of {valid_side_uses}. Passed: {side}")

    # ======
    # CREATE FULL SINC
    # ======
    # Create the full sinc function as normal
    if return_gz:
        rf, gz, gzr = pp.make_sinc_pulse(flip_angle=flip_angle, apodization=apodization, delay=delay, duration=duration*2,
                                         dwell=dwell, center_pos=center_pos, freq_offset=freq_offset, max_grad=max_grad,
                                         max_slew=max_slew, phase_offset=phase_offset, return_gz=return_gz,
                                         slice_thickness=slice_thickness, system=system,
                                         time_bw_product=time_bw_product, use=use)
    else:
        rf = pp.make_sinc_pulse(flip_angle=flip_angle, apodization=apodization, delay=delay, duration=duration*2,
                                dwell=dwell, center_pos=center_pos, freq_offset=freq_offset, max_grad=max_grad,
                                max_slew=max_slew, phase_offset=phase_offset, return_gz=return_gz,
                                slice_thickness=slice_thickness, system=system, time_bw_product=time_bw_product,
                                use=use)

    # ======
    # MODIFY SINC
    # ======
    # Modify the sinc pulse to be a half sinc

    if len(rf.signal) % 2 != 0:
        raise ValueError("The signal array has an odd number of elements.")

    half_length = int(len(rf.signal) / 2)
    rf.t = rf.t[:half_length]
    if side == "left":
        rf.signal = rf.signal[:half_length]
    if side == "right":
        rf.signal = rf.signal[half_length:]

    if dwell == 0:
        dwell = system.rf_raster_time
    rf.shape_dur = np.float64(half_length * dwell)  # correct the shape duration

    # ======
    # RETURN Modified
    # ======
    # Return the half sinc pulse
    if return_gz:
        return rf, gz, gzr
    return rf
