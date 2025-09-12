import pypulseq as pp
from typing import Tuple, Union
from types import SimpleNamespace
from pypulseq.convert import convert


def make_basic_gx_gradient_separat(system: Union[pp.Opts, None] = None,
                                   slice_profile: bool=False,
                                   gx_amp: float = -4.7,
                                   gx_rut: float = 30e-6,
                                   gx_flat: float = 640e-6,
                                   gx_rdt: float = 90e-6,
                                   spoil_amp: float = -11.74,
                                   spoil_rut: float = 90e-6,
                                   spoil_flat: float = 120e-6,
                                   spoil_rdt: float = 70e-6,
                                   delay: float = 30e-6
                                   )-> Union[
                                            Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace],
                                            Tuple[SimpleNamespace, SimpleNamespace],
                                            ]:
    """
    Creates a readout gradient with spoiler.
    If `slice_profile` is set to True, it creates a gradient with prephaser for a slice profile sequence

    Parameters
    --------
    system : Opts, default=Opts()
        System limits. Default is a system limits object initialized to default values.
    slice_profile : bool
        Boolean flag to indicate, if a gradient for a slice profile sequence is to be returned.
    gx_amp : float
        Amplitude of read-out gradient in mT/m.
    gx_rut : float
        Ramp up/ rise time of read-out gradient in seconds (s).
    gx_flat : float
        Flat time of read-out gradient in seconds (s).
    gx_rdt : float
        Ramp down/ fall time of read-out gradient.
    spoil_amp : float
        Amplitude of spoiler in mT/m.
    spoil_rut : float
        Ramp up/ rise time of spoiler in seconds (s).
    spoil_flat : float
        Flat time of spoiler in seconds (s).
    spoil_rdt : float
        Ramp down/ fall time of spoiler in seconds (s).
    delay : float
        Delays gradients in regard to ADC in seconds (s).

    Returns
    --------
    gx : SimpleNamespace
        Read-out gradient
    spoil : SimpleNamespace
        Spoiler
    prephase : SimpleNamespace, optional
        Prephaser Gradient for read-out. Returned only if `slice_profile` is set to True.
    """

    gx_amp = convert(from_value=gx_amp, from_unit='mT/m', to_unit='Hz/m')  # Hz/m
    spoil_amp = convert(from_value=spoil_amp, from_unit='mT/m', to_unit='Hz/m')  # Hz/m

    if slice_profile:
        channel = 'z'
        gx = pp.make_trapezoid(channel=channel, amplitude=gx_amp, rise_time=gx_rut, flat_time=gx_flat, fall_time=gx_rdt,
                               system=system)
        prephase = pp.make_trapezoid(channel=channel, area=-gx.area/2, system=system, duration=max(pp.calc_duration(gx)/4, 200e-6))
        # NotImplementedError: Amplitude + Area input pair is not implemented yet.
        gx.delay = prephase.rise_time + prephase.flat_time + prephase.fall_time
        spoil = pp.make_trapezoid(channel=channel, amplitude=spoil_amp, rise_time=spoil_rut, flat_time=spoil_flat,
                                  fall_time=spoil_rdt,
                                  delay=gx_rut + gx_flat+prephase.rise_time+prephase.flat_time+prephase.fall_time, system=system)
        return gx, spoil, prephase
    else:
        channel = 'x'

        gx = pp.make_trapezoid(channel=channel, amplitude=gx_amp, rise_time=gx_rut, flat_time=gx_flat, fall_time=gx_rdt,
                                   delay=delay, system=system)
        spoil = pp.make_trapezoid(channel=channel, amplitude=spoil_amp, rise_time=spoil_rut, flat_time=spoil_flat,
                                  fall_time=spoil_rdt,
                                  delay=gx_rut + gx_flat + delay, system=system)
        return gx, spoil
