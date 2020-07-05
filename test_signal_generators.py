###############################################################################
# Project name: Test signal generators                                        #
# Author: Joao Nuno Carvalho                                                  #
# Date:   2020.07.04                                                          #
#                                                                             #
# Description: This is an implementation in Python of algorithms thatgenerate #
#              several different test signals.                                #
# You can find the following test signal implementations:                     #
#                                                                             #
#   1. Simple sine or cosine functions with phase.                            #
#   2. Two crafted sinusoids combined.                                        #
#   3. N crafted sinusoids combined.                                          #
#   4. Chirp from freq A to freq B can also be called a continues             #
#      frequency sweep.                                                       #
#   5. One Dirac impulse.                                                     #
#   6. One step cycle function.                                               #
#   7. One square cycle function.                                             #
#   8. Square wave function.                                                  #
#   9. Uniform noise from freq A to freq B.                                   #
#                                                                             #
#   The spectrum amplitude of each signal is also showed.                     #
#                                                                             #
# License: MIT Opensource License.                                            #
#                                                                             #
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

##########
# 1. Sine Signal

def generate_sine(sample_rate, signal_duration, freq, phase, amplitude):
    freq      = float(freq)
    phase     = float(phase)
    amplitude = float(amplitude)
    start = 0.0
    stop = signal_duration
    step = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    s = amplitude*np.sin(phase + 2*np.pi*freq*t)
    len_s = s.size
    return (t, s, len_s)

def plot_sine(t, s, freq, phase):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Sine signal freq. ' + str(freq) + ' Hz' + ' phase ' + "{0:.2f}".format(phase) + ' Rad'  )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_sine_signal():
    # Generate sine 1.
    sample_rate = 40000
    freq        = 500.0
    phase       = 0.0
    amplitude   = 1.0
    signal_duration = 0.01
    t, s, len_s = generate_sine(sample_rate, signal_duration, freq, phase, amplitude)
    plot_sine(t, s, freq, phase)

    # Generate sine 2.
    sample_rate = 40000
    freq        = 500.0
    phase       = np.pi / 2.0
    amplitude   = 1.0
    signal_duration = 0.01
    t, s, len_s = generate_sine(sample_rate, signal_duration, freq, phase, amplitude)
    plot_sine(t, s, freq, phase)
    plot_spectrum(t, s, sample_rate)

#########
# 2. Two crafted sinusoids combined.

def generate_2_sines(sample_rate, signal_duration, freq_1, phase_1, amplitude_1, freq_2, phase_2, amplitude_2):
    freq_1      = float(freq_1)
    phase_1     = float(phase_1)
    amplitude_1 = float(amplitude_1)
    freq_2      = float(freq_2)
    phase_2     = float(phase_2)
    amplitude_2 = float(amplitude_2)
    t_1, s_1, len_s_1 = generate_sine(sample_rate, signal_duration, freq_1, phase_1, amplitude_1)
    t_2, s_2, len_s_2 = generate_sine(sample_rate, signal_duration, freq_2, phase_2, amplitude_2)
    s_output = (s_1 + s_2) / 2.0
    len_s = s_output.size
    return (t_1, s_output, len_s)

def plot_2_sines(t, s, freq_1, phase_1, freq_2, phase_2 ):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Two sines signals freq_a ' + str(freq_1) + ' Hz' + ' phase_a ' + "{0:.2f}".format(phase_1) + ' Rad' +
              ' freq_b ' + str(freq_2) + ' Hz' + ' phase_b ' + "{0:.2f}".format(phase_2) + ' Rad' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_2_sines_signal():
    sample_rate = 40000
    freq_1      = 500.0
    phase_1     = 0.0
    amplitude_1 = 1.0
    freq_2      = 750.0
    phase_2     = np.pi*(2/3.0)
    amplitude_2 = 1.0
    signal_duration = 0.01
    t, s, len_s = generate_2_sines(sample_rate, signal_duration, freq_1, phase_1, amplitude_1,
                                   freq_2, phase_2, amplitude_2)
    plot_2_sines(t, s, freq_1, phase_1, freq_2, phase_2)
    plot_spectrum(t, s, sample_rate)

#########
# 3. N crafted sinusoids combined.

def generate_N_sines(sample_rate, signal_duration, freq_list, phase_list, amplitude_list):
    len_a = len(freq_list)
    len_b = len(phase_list)
    len_c = len(amplitude_list)
    if ((len_a != len_b != len_c) and len_a > 0 and len_b > 0 and len_c > 0):
        print("ERROR: In function generate_N_sines() parameters freq_list, phase_list and amplitude_list are not of equal size and greater then zero.")
        exit(-1)
    t_output = None
    accumulator = None
    len_seq = len(freq_list)
    for i in range(len_seq):
        freq      = float(freq_list[i])
        phase     = float(phase_list[i])
        amplitude = float(amplitude_list[i]) 
        t, s, len_s = generate_sine(sample_rate, signal_duration, freq, phase, amplitude)
        if t_output is None:
            t_output = t
        accumulator = s if (accumulator is None) else accumulator + s
    s_output = accumulator / float(len_seq) 
    len_s = s_output.size
    return (t_output, s_output, len_s)

def plot_N_sines(t, s, freq_list, phase_list):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    str_freq_phase = ["freq {0:.2f} Hz, phase {1:.2f} Rad".format(pair[0], pair[1]) for pair in zip(freq_list, phase_list)]    
    plt.title('N sines ' + "".join(str_freq_phase))
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_N_sines_signal():
    sample_rate     = 40000
    freq_list       = [500.0, 750.0, 950.0]
    phase_list      = [0.0, np.pi/2.0, np.pi*(2/3)]
    amplitude_list  = [1.0, 1.0, 1.0]
    signal_duration = 0.01
    t, s, len_s = generate_N_sines(sample_rate, signal_duration, freq_list, phase_list, amplitude_list)
    plot_N_sines(t, s, freq_list, phase_list)
    plot_spectrum(t, s, sample_rate)

##########
# 4. Chirp Signal

def generate_chirp(sample_rate, chirp_duration, start_freq, end_freq):
    f_0 = start_freq
    f_1 = end_freq
    start = 0.0
    stop = chirp_duration
    step = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    phase = 0.0
    chirp_period = chirp_duration # 1 / 100.0 #1.0
    k = (f_1 - f_0) / chirp_period
    s = np.sin(phase + 2*np.pi * ( f_0*t + (k/2)*np.square(t)) )
    len_s = s.size
    return (t, s, len_s)

def plot_chirp(t, s, start_freq, end_freq):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Chirp  from ' + str(start_freq) + 'Hz  to ' + str(end_freq) + 'Hz' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_chirp_signal():
    sample_rate = 40000
    start_freq = 500
    end_freq   = 5000
    # chirp_duration = 1.00
    chirp_duration = 0.01
    t, s, len_s = generate_chirp(sample_rate, chirp_duration, start_freq, end_freq)
    plot_chirp(t, s, start_freq, end_freq)
    plot_spectrum(t, s, sample_rate)

##########
# 5. One Dirac impulse.

def generate_dirac_impulse(sample_rate, buffer_duration, dirac_impulse_offset):
    start = 0.0
    stop = buffer_duration
    step = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    s = np.zeros(len(t), dtype='float')
    s[dirac_impulse_offset] = 1.0
    len_s = s.size
    return (t, s, len_s)

def plot_dirac_impulse(t, s, dirac_impulse_offset):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Dirac impulse offset ' + str(dirac_impulse_offset) + ' samples' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_dirac_impulse_signal():
    sample_rate = 40000
    # chirp_duration = 1.00
    buffer_duration = 0.01
    dirac_impulse_offset = 20  # Samples
    t, s, len_s = generate_dirac_impulse(sample_rate, buffer_duration, dirac_impulse_offset)
    plot_dirac_impulse(t, s, dirac_impulse_offset)
    plot_spectrum(t, s, sample_rate)

###########
# 6. One step function.

def generate_step(sample_rate, buffer_duration, step_impulse_offset):
    start = 0.0
    stop  = buffer_duration
    step  = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    s = np.zeros(len(t), dtype='float')
    for i in range(0, len(s)):
        if i >= step_impulse_offset:
            s[i] = 1
    len_s = s.size
    return (t, s, len_s)

def plot_step(t, s, step_impulse_offset):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('One step impulse offset ' + str(step_impulse_offset) + ' samples' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_step_signal():
    sample_rate = 40000
    # chirp_duration = 1.00
    buffer_duration = 0.01
    step_impulse_offset = 20  # Samples
    t, s, len_s = generate_step(sample_rate, buffer_duration, step_impulse_offset)
    plot_step(t, s, step_impulse_offset)
    plot_spectrum(t, s, sample_rate)

###########
# 7. One square function.

def generate_square(sample_rate, buffer_duration, square_impulse_offset):
    start = 0.0
    stop  = buffer_duration
    step  = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    s = np.zeros(len(t), dtype='float')
    for i in range(0, len(s)):
        if (i >= square_impulse_offset and i <= 2*square_impulse_offset ):
            s[i] = 1
    len_s = s.size
    return (t, s, len_s)

def plot_square(t, s, square_impulse_offset):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Square impulse offset_1 ' + str(square_impulse_offset) + ', samples offset_2 ' +
              str(2*square_impulse_offset+1) + ' samples' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_square_signal():
    sample_rate           = 40000
    buffer_duration       = 0.01
    square_impulse_offset = 20  # Samples
    t, s, len_s = generate_square(sample_rate, buffer_duration, square_impulse_offset)
    plot_square(t, s, square_impulse_offset)
    plot_spectrum(t, s, sample_rate)

#########
# 8. Square wave function.

def generate_square_wave(sample_rate, buffer_duration, square_wave_width):
    start = 0.0
    stop  = buffer_duration
    step  = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    s = np.zeros(len(t), dtype='float')
    counter = square_wave_width
    flag_val = False 
    for i in range(0, len(s)):
        if counter == 0:
            flag_val = not flag_val
            counter = square_wave_width
        s[i] = 0.0 if (flag_val == False) else 1.0
        counter -= 1
    len_s = s.size
    return (t, s, len_s)

def plot_square_wave(t, s, square_wave_width):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Square wave width ' + str(square_wave_width) + ' samples' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_square_wave_signal():
    sample_rate      = 40000
    buffer_duration  = 0.01   # 1
    square_wave_width = 20    # Samples
    t, s, len_s = generate_square_wave(sample_rate, buffer_duration, square_wave_width)
    plot_square_wave(t, s, square_wave_width)
    plot_spectrum(t, s, sample_rate)

#########
# 9. Uniform noise from freq A to freq B.

def generate_noise(sample_rate, signal_duration, start_freq, end_freq):
    # We use the max sample_rate for the fft sample size, so that all
    # frequencies up until the max sample rate can be used.
    # The generated signal duration is always one second, with 
    # the buffer size of the sample rate.

    # Note: Inspired on the post of stackoverflow:
    #       How to generate noise in frequency range with numpy?
    # https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy

    # In here we determine the frequencies for sample_rate or fft buffer_size.
    fft_freq = np.abs(np.fft.fftfreq(sample_rate, 1 / sample_rate))
 
    chosen_freq_bins = np.zeros(sample_rate)
    for i in range(0, len(fft_freq)):
        if (fft_freq[i] >= start_freq and fft_freq[i] <= end_freq):
            chosen_freq_bins[i] = int(1)
    ifft_buffer = np.array(chosen_freq_bins, dtype=complex)
    N_positive = (len(ifft_buffer) - 1) // 2
    # Note: The amplitude is always 1 (one), the phase is what can change
    #       randomly.
    # Phase in the interval [0, 2*Pi].
    phase_value = np.random.rand(N_positive) * 2 * np.pi
    complex_value = np.cos(phase_value) + 1j * np.sin(phase_value)
    # The first half of the buffer has the phase.
    ifft_buffer[ 1: N_positive + 1] *= complex_value
    # The second half of the buffer has complex conjugate of the phase. 
    ifft_buffer[ -1 : -1-N_positive : -1] = np.conj(ifft_buffer[ 1: N_positive + 1])
    # Get the real part of the iFFT of the input buffer for the generated
    # noise for the selected frequencies.
    output_buffer = np.fft.ifft(ifft_buffer).real
    
    if signal_duration > 1:
        signal_duration = 1.0    # Max. 1 second
    start = 0.0
    stop = signal_duration     
    step = 1.0 / sample_rate
    t = np.arange(start, stop, step, dtype='float')
    output_buffer = output_buffer[0 : len(t)]
    return (t, output_buffer, len(output_buffer))

def plot_noise(t, s, start_freq, end_freq):
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('Noise start freq ' + str(start_freq) + ' Hz, stop freq. ' +
              str(end_freq) + ' Hz' )
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

def test_noise_signal():
    sample_rate = 40000
    start_freq  = 500.0  
    stop_freq   = 750.0
    signal_duration = 0.01
    t, s, len_s = generate_noise(sample_rate, signal_duration, start_freq, stop_freq)
    plot_noise(t, s, start_freq, stop_freq)
    plot_spectrum(t, s, sample_rate)

########
# FFT Spectrum Analysis.

def plot_spectrum(t, s, sample_rate):
    if len(s) > sample_rate:
        s = s[0 : sample_rate - 1]

    realFFT = np.abs(np.fft.rfft(s))
    for i in range(0, len(realFFT)):
        if realFFT[i] == 0.0:
            realFFT[i] = 1e-20
    output_buffer_magnitude = 10*np.log10(realFFT)
    fft_freq = np.abs(np.fft.fftfreq(len(s), 1 / sample_rate))
    fft_freq = fft_freq[0 : int((len(s) / 2) + 1)]

    # # Artificial limit, only for DEBUG.
    # fft_freq = fft_freq[0 : int(len(fft_freq) / 8)]
    # output_buffer_magnitude = output_buffer_magnitude[0 : int(len(output_buffer_magnitude) / 8)]

    plt.plot(fft_freq, output_buffer_magnitude)
    
    plt.xlabel('freq (Hz)')
    plt.ylabel('amplitude')
    plt.title('FFT Spectrum')
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()

if __name__ == "__main__":
    test_sine_signal()
    test_2_sines_signal()
    test_N_sines_signal()
    test_chirp_signal()
    test_dirac_impulse_signal()
    test_step_signal()
    test_square_signal()
    test_square_wave_signal()
    test_noise_signal()
    



