import numpy as np
import pypulseq as pp
from typing import Tuple, Union
from types import SimpleNamespace
from pypulseq.convert import convert


def make_z_gradient_separat(system: Union[pp.Opts, None] = None,
                            ss_amp: float = 5.87,
                            ss_rut: float = 100e-6,
                            ss_flat: float = 400e-6,
                            ss_rdt: float = 30e-6,
                            mp_amp: float = -18.81,
                            mp_rut: float = 100e-6,
                            mp_flat: float = 170e-6,
                            mp_rdt: float = 100e-6,
                            rp_amp: float = -2.94,
                            rp_rut: float = 30e-6,
                            rp_rdt: float = 30e-6
                            ) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """
    Create the specific slice select gradient for imaging with double-half pulses.

    Parameters
    --------
    system : Opts, default=Opts()
        System limits. Default is a system limits object initialized to default values.
    ss_amp : float
        Amplitude of the slice select gradient (first and third part) in mT/m.
    ss_rut : float
        Ramp up/ rise time of slice select gradient in seconds (s).
    ss_flat : float
        Flat time of slice select gradient in seconds (s).
    ss_rdt: float
        Ramp down/ fall time of slice select gradient in seconds (s).
    mp_amp: float
        Amplitude of the mid phase gradient (second part) in mT/m.
    mp_rut: float
        Ramp up/ rise time of mid phase gradient in seconds (s).
    mp_flat: float
        Flat time of mid phase gradient in seconds (s).
    mp_rdt: float
        Ramp down/ fall time of mid phase gradient in seconds (s).
    rp_amp: float
        Amplitude of the rephase gradient (fourth part) in mT/m.
    rp_rut: float
        Ramp up/ rise time of the rephase gradient in seconds (s).
    rp_rdt: float
        Ramp down/ fall time of the rephase gradient in seconds (s).

    Returns
    --------
    slice_select_extended : extended_trapezoid
        first and third part of the whole slice select gradient
    mid_phase_extended : extended_trapezoid
        second part of the whole slice select gradient
    rephase_extended : extended_trapezoid
        forth part of the whole slice select gradient
    """
    # Convert the amplitudes to Hz/m for use in the make (extended) trapezoid function
    ss_amp = convert(from_value=ss_amp, from_unit='mT/m', to_unit='Hz/m')  # Hz/m
    mp_amp = convert(from_value=mp_amp, from_unit='mT/m', to_unit='Hz/m')  # Hz/m
    rp_amp = convert(from_value=rp_amp, from_unit='mT/m', to_unit='Hz/m')  # Hz/m

    # Base points for slice select gradient.
    t_slice_select_extended = np.array([
        0,
        ss_rut,
        ss_rut + ss_flat,
        ss_rut + ss_flat + ss_rdt

    ])
    amp_slice_select_extended = np.array([
        0,
        ss_amp,
        ss_amp,
        0
    ])

    # Base points for mid-phase gradient.
    t_mid_phase_extended = np.array([
        0,
        mp_rut,
        mp_rut + mp_flat,
        mp_rut + mp_flat + mp_rdt
    ])
    amp_mid_phase_extended = np.array([
        0,
        mp_amp,
        mp_amp,
        0
    ])

    # Base points for rephase gradient.
    t_rephase_extended = np.array([
        0,
        rp_rut,
        rp_rut + rp_rdt
    ])
    amp_rephase_extended = np.array([
        0,
        rp_amp,
        0
    ])

    # Make extended trapezoid forms.
    slice_select_extended = pp.make_extended_trapezoid(channel='z', times=t_slice_select_extended,
                                                       amplitudes=amp_slice_select_extended, system=system)
    mid_phase_extended = pp.make_extended_trapezoid(channel='z', times=t_mid_phase_extended,
                                                    amplitudes=amp_mid_phase_extended, system=system)
    rephase_extended = pp.make_extended_trapezoid(channel='z', times=t_rephase_extended,
                                                  amplitudes=amp_rephase_extended, system=system)
    # print(f"Are of slice select: {slice_select_extended.area:.2f}")
    # print(f"Are of midphase: {mid_phase_extended.area:.2f}")
    # print(f"Are of rephase: {rephase_extended.area:.2f}")

    return slice_select_extended, mid_phase_extended, rephase_extended
