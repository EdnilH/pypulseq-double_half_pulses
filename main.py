import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from datetime import datetime
from pypulseq.utils.safe_pns_prediction import safe_example_hw
from warnings import warn
from make_z_gradient import make_z_gradient_separat
from make_ro_gradient import make_basic_gx_gradient_separat
from make_half_sinc_pulse import make_half_sinc_pulse


def main(plot_seq=True, plot_k_space_traj=True, plot_2D_k_space=True, plot_grad=False, write_seq=False,
         seq_filename: str = 'double_half_pulse_pypulseq', save=False, slice_profile=False):

    double_golden_angle = 4 * np.pi / (np.sqrt(5) + 1)  # Angular increment

    # Set system limits
    system = pp.Opts(max_grad=58,
                     grad_unit='mT/m',
                     # grad_raster_time=10e-6,  # standard
                     max_slew=200,  # 180
                     slew_unit='mT/m/ms',
                     rf_dead_time=100e-6,
                     # rf_raster_time=1e-6,  # standard
                     # adc_raster_time=100e-9,  #standard
                     rf_ringdown_time=0,
                     B0=2.893620,
                     gamma=42.577e6
                     )

    seq = pp.Sequence(system=system)  # Create a new sequence object, with system settings

    num_spokes = 4  # Number of radial spokes
    num_samples = 128  # Number of measurement points in adc
    flip_angle = 7 # degree
    rf_duration = 400e-6 # seconds
    rf_time_bw_product = 2  # 2*Number of zeros in sinc
    rf_delay = system.rf_dead_time
    rf_apodization = 0.5  # ?

    # ======
    # Prepare Events
    # ======
    slice_select, mid_phase, rephase = make_z_gradient_separat(system=system, ss_rut=rf_delay)
    if not slice_profile:
        gx_base, spoil_base = make_basic_gx_gradient_separat(system=system, delay=60e-6)
        gx_and_spoil = pp.add_gradients([gx_base, spoil_base])
        adc = pp.make_adc(num_samples=num_samples, duration=gx_base.flat_time, delay=0, system=system)
    else:
        gx_base, spoil_base, prephase_base = make_basic_gx_gradient_separat(system=system, delay=0, slice_profile=True)
        gx_and_spoil = pp.add_gradients([prephase_base, gx_base, spoil_base])
        adc = pp.make_adc(num_samples=num_samples, duration=gx_base.flat_time,
                          delay=pp.calc_duration(prephase_base)+gx_base.rise_time, system=system)

    rf_left = make_half_sinc_pulse(flip_angle=flip_angle * np.pi / 180,
                                   side="left",
                                   apodization=rf_apodization,
                                   duration=rf_duration,
                                   delay=rf_delay,  # rf on flat of slice select
                                   system=system,
                                   use='excitation',
                                   time_bw_product=rf_time_bw_product)
    rf_right = make_half_sinc_pulse(flip_angle=flip_angle * np.pi / 180,
                                    side="right",
                                    apodization=rf_apodization,
                                    duration=rf_duration,
                                    delay=rf_delay,
                                    system=system,
                                    use='excitation',
                                    time_bw_product=rf_time_bw_product)

    # ======
    # Create Sequence
    # ======
    for i in range(0, num_spokes):
        # Add complete slice select with rf pulses at corresponding times.
        seq.add_block(slice_select, rf_right)
        seq.add_block(mid_phase)
        seq.add_block(slice_select, rf_left)
        seq.add_block(rephase)
        # Readout gradient gets rotated by the double golden angle for each spoke (i).
        # If Gradient is purely on x or y axis, the function returns only one parameter, otherwise two.
        # This behaviour is caught here.
        if len(pp.rotate(gx_and_spoil, angle=i*double_golden_angle, axis='z')) == 1:
            g = pp.rotate(gx_and_spoil, angle=i*double_golden_angle, axis='z')[0]
            seq.add_block(g, adc)
        else:
            gx, gy = pp.rotate(gx_and_spoil, angle=i*double_golden_angle, axis='z')
            seq.add_block(gx, gy, adc)
        if i == 0:
            # The length of one complete block should be TR.
            print(f"TR: {seq.duration()[0]*10**3:.2f}ms")

    print(f"Duration of entire sequence: {seq.duration()[0]*1e3:.2f}ms")

    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    ## PNS calc (Peripheral Nerve Stimulation)
    pns_ok, pns_norm, pns_components, time_axis_pns = seq.calculate_pns(safe_example_hw(), do_plots=plot_seq)  # Safe example HW
    if pns_ok:
        print(f'PNS check passed successfully with max {max(pns_norm)*100:.2f}%')
    else:
        warn(f'PNS check failed with max {max(pns_norm)*100:.2f}%')

    # ======
    # VISUALIZATION
    # ======
    folder_plots = "/data/projects/mptrainee_pulseq/media/"
    prefix = f"{datetime.now().strftime('%Y%m%d-%H_%M')}-{num_spokes}_spokes-{num_samples}_samples"
    if plot_seq:
        seq.plot(time_disp='ms', plot_now=False, show_blocks=False, save=save)

    # Trajectory calculation
    ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
    last_kx_t = time_axis[-1]
    last_adc_t = t_adc[-1]
    fac = last_kx_t / (last_adc_t + spoil_base.flat_time + spoil_base.fall_time + 30e-6)
    time_axis = time_axis / fac

    if plot_k_space_traj:
        plt.figure()
        plt.title("k-space components as functions of time")
        plt.xlabel("time [s]")
        plt.ylabel("kx, ky, kz")
        plt.plot(time_axis, ktraj[0].T, label='x')
        plt.plot(time_axis, ktraj[1].T, label='y')
        plt.plot(time_axis, ktraj[2].T, label='z')  # Plot the entire k-space trajectory
        plt.plot(t_adc, ktraj_adc[0], '.', label='adc x')  # Plot sampling points on the kx-axis
        plt.plot(t_adc, ktraj_adc[1], '.', label='adc y')  # Plot sampling points on the ky-axis
        plt.legend()
        if save:
            plt.savefig(f"{folder_plots}{prefix}-trajectory.png")

    if plot_2D_k_space:
        plt.figure()
        plt.title('2D k-space')
        plt.xlabel("kx")
        plt.ylabel("ky")
        plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
        plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')  # Plot  sampling points
        plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
        if save:
            plt.savefig(f"{folder_plots}{prefix}-2D_k_space.png")

    if plot_grad:
        # Plot gradients to check for gaps and optimality of the timing
        gw = seq.waveforms_and_times()[0]
        # Plot the entire gradient shape
        plt.figure()
        plt.title('Entire Gradient Shape')
        plt.xlabel('time [s]')
        plt.ylabel('Hz/m')
        plt.plot(gw[0][0], gw[0][1], gw[1][0], gw[1][1], gw[2][0], gw[2][1])
        if save:
            plt.savefig(f"{folder_plots}{prefix}-gradient_shape.png")

    if plot_seq or plot_k_space_traj or plot_2D_k_space or plot_grad:
        plt.show()

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        # seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
        seq.set_definition(key='Name', value='UTE-Double_Half_Sinc-Hannah')
        folder_seq = "/data/projects/mptrainee_pulseq/sequences/"
        if slice_profile:
            seq.write(f"{folder_seq}slice_profile_sequence-{num_spokes}_spokes-{num_samples}-inverse_slice_select_gradient.seq")
        else:
            seq.write(f"{folder_seq}{seq_filename}-{num_spokes}_spokes-{num_samples}_samples-inverse_slice_select_gradient.seq")
            np.savetxt(f"{prefix}-trajectory.txt",ktraj_adc ,delimiter=";",fmt="%g")


if __name__ == '__main__':
    main(plot_seq=True,
         plot_2D_k_space=False,
         plot_k_space_traj=False,
         plot_grad=False,
         write_seq=False,
         save=False,
         slice_profile=False
         )
