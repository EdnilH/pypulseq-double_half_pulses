import math
from copy import copy
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_sinc_pulse import make_sinc_pulse


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
                         system: Union[Opts, None] = None,
                         time_bw_product: float = 4,
                         use: str = str()
                         ) -> Union[
                                    SimpleNamespace,
                                    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace],
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
        gzm : SimpleNamespace, optional
           Accompanying slice select mid-phase trapezoidal gradient event. Returned only if `slice_thickness` is provided.
        gzr : SimpleNamespace, optional
           Accompanying slice select rephasing trapezoidal gradient event. Returned only if `slice_thickness` is provided.

        Raises
        ------
        ValueError
           If invalid `side` parameter was passed. Must be one of 'left' or 'right'
           If length of the full sinc pulse is uneven.
           If `return_gz=True` and `slice_thickness` was not provided.
        """
    valid_side_uses = ["left", "right"]
    if side not in valid_side_uses:
        raise ValueError(f"Invalid side parameter. Must be one of {valid_side_uses}. Passed: {side}")

    # ======
    # CREATE FULL SINC
    # ======
    # Create the full sinc function as normal
    rf = make_sinc_pulse(flip_angle=flip_angle, apodization=apodization, delay=delay, duration=duration*2,
                            dwell=dwell, center_pos=center_pos, freq_offset=freq_offset,
                            phase_offset=phase_offset, return_gz=False,
                            system=system, time_bw_product=time_bw_product, use=use)
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

    # ======
    # CREATE SLICE SELECT GRADIENT
    # ======
    # Create the specific slice select gradient for imaging with (double)-half pulses
    if return_gz:
        if slice_thickness == 0:
            raise ValueError('Slice thickness must be provided')
        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        area = (time_bw_product/2) / slice_thickness  # only half a pulse, so the tbw is only half as long

        gz = make_trapezoid(channel='z', system=system, flat_time = duration, flat_area=area)
        print(f"Are of slice select: {gz.area:.2f}")
        gzm = make_trapezoid(channel='z', system=system, area = -(gz.area + gz.flat_area + 0.5 * gz.rise_time * gz.amplitude))
        print(f"Are of midphase: {gzm.area:.2f}")
        gzr = make_trapezoid(channel='z', system=system, area = -(0.5 * gz.fall_time * gz.amplitude))
        print(f"Are of rephase: {gzr.area:.2f}")
        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    # correct the shape duration
    if dwell == 0:
        dwell = system.rf_raster_time
    rf.shape_dur = np.float64(half_length * dwell)

    # ======
    # RETURN Modified
    # ======
    # Return the half sinc pulse
    if return_gz:
        return rf, gz, gzm, gzr
    return rf
