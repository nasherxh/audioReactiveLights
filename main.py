#visualisation imports
from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import pyaudio
import microphone
import dsp
import led
import sys
_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""
_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""
uniquePixels = config.N_PIXELS


#main function imports
#import time
import RPi.GPIO as GPIO
import signal
#import sys

import colourwheel
#import config
import board
import neopixel
pixels= neopixel.NeoPixel(board.D18, config.N_PIXELS,auto_write=False)

#import turn off LEDs
import off

#set warnings off (optional)
GPIO.setwarnings(False)
# Set up GPIO to use Broadcom GPIO numbers
GPIO.setmode(GPIO.BCM) 
#set Button pin
BUTTON_GPIO =16
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#setCycleCount
cycleCount=0;



def frames_per_second():
    """Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


def createFilters():
    global r_filt
    global g_filt
    global b_filt
    global common_mode
    global p
    global p_filt
    global uniquePixels

    r_filt = dsp.ExpFilter(np.tile(0.01, uniquePixels),
                             alpha_decay=0.2, alpha_rise=0.99)
    g_filt = dsp.ExpFilter(np.tile(0.01, uniquePixels),
                             alpha_decay=0.05, alpha_rise=0.3)
    b_filt = dsp.ExpFilter(np.tile(0.01, uniquePixels),
                             alpha_decay=0.1, alpha_rise=0.5)
    common_mode = dsp.ExpFilter(np.tile(0.01, uniquePixels),
                             alpha_decay=0.99, alpha_rise=0.01)
    p_filt = dsp.ExpFilter(np.tile(1, (3, uniquePixels)),
                             alpha_decay=0.1, alpha_rise=0.99)
    p = np.tile(1.0, (3, uniquePixels))

gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *=0.89
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Update the LED strip
    return np.concatenate((p, p[:, ::-1], p, p[:, ::-1] ), axis=1)
    
def visualize_scroll2(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.94
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Update the LED strip
    return np.concatenate((p, p ), axis=1)


def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float(uniquePixels - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.92
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p, p[:, ::-1], p, p[:, ::-1]), axis=1)

def visualize_energy2(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float(uniquePixels - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.97
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p, p), axis=1)
    
_prev_spectrum = np.tile(0.01, uniquePixels)


def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum
    y = np.copy(interpolate(y, uniquePixels))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))
    # Mirror the color channels for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    output = np.array([r, g,b]) * 255
    return output


fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    # Normalize samples between 0 and 1
    y = audio_samples / 2.0**15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    
    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transform audio input into the frequency domain
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
        if config.USE_GUI:
            # Plot filterbank output
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            mel_curve.setData(x=x, y=fft_plot_filter.update(mel))
            # Plot the color channels
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
    if config.USE_GUI:
        app.processEvents()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))


# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

def start_stream(callback,cycleCountInput):
    p = pyaudio.PyAudio()
    frames_per_buffer = int(config.MIC_RATE / config.FPS)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=config.MIC_RATE,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    overflows = 0
    prev_ovf_time = time.time()
    while True:
        if cycleCount!=cycleCountInput:
            print("cycleCountChanged")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return()
        try:
            y = np.fromstring(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
            y = y.astype(np.float32)
            stream.read(stream.get_read_available(), exception_on_overflow=False)
            callback(y)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()


def signal_handler(sig, frame):
	GPIO.cleanup()
	sys.exit(0)

def button_pressed_callback(channel):
	print("Button pressed!")
	global cycleCount
	cycleCount =(cycleCount+1)%10
	print(cycleCount)


def rainbowcycle(wait):
	print("Rainbow cycling")
	global cycleCount
	for j in range(255):
		for i in range(config.N_PIXELS):
			if cycleCount !=1:
				print("cycleCountChanged")
				return()
				
			#take each pixel 'i' as a fraction of the full colour wheel 
			#by dividing by number of pixels and multiplying by colour wheel spectrum (256).
			#Take integer division discarding remainder (//).
			#This will assign each pixel one colour in the whole colour wheel spectrum
			#Add j to make each pixel slowly cycle through the colour wheel individually (as j increases up to 255)
			pixel_index = (i * 256 // config.N_PIXELS) + j 
			#Write to pixel 'i'. Take modulo so only write a value between 0-255
			pixels[i] = colourwheel.wheel(pixel_index % 256)
		pixels.show()
		time.sleep(wait)
	print("finished cycle");

if __name__ == "__main__":
	GPIO.add_event_detect(BUTTON_GPIO, GPIO.FALLING, 
			callback=button_pressed_callback, bouncetime=200)
	while(True):
		if cycleCount ==0:
			off.LEDsOff()
		elif cycleCount==1:
			uniquePixels = config.N_PIXELS
			rainbowcycle(0.01)
		elif cycleCount==2:
			visualization_type = visualize_scroll
			visualization_effect = visualization_type
			uniquePixels = config.N_PIXELS // 4
			createFilters()
			"""Visualization effect to display on the LED strip"""
			# Initialize LEDs
			led.update()
			# Start listening to live audio stream
			start_stream(microphone_update,2)
		elif cycleCount==3:
			visualization_type = visualize_scroll2
			visualization_effect = visualization_type
			uniquePixels = config.N_PIXELS // 2
			createFilters()
			"""Visualization effect to display on the LED strip"""
			# Initialize LEDs
			led.update()
			# Start listening to live audio stream
			start_stream(microphone_update,cycleCount)
		elif cycleCount==4:
			visualization_type = visualize_energy
			visualization_effect = visualization_type
			uniquePixels = config.N_PIXELS //4
			createFilters()
			"""Visualization effect to display on the LED strip"""
			# Initialize LEDs
			led.update()
			# Start listening to live audio stream
			start_stream(microphone_update,4)
		elif cycleCount==5:
			visualization_type = visualize_energy2
			visualization_effect = visualization_type
			uniquePixels = config.N_PIXELS //2
			createFilters()
			"""Visualization effect to display on the LED strip"""
			# Initialize LEDs
			led.update()
			# Start listening to live audio stream
			start_stream(microphone_update,5)
		elif cycleCount==6:
			visualization_type = visualize_spectrum
			visualization_effect = visualization_type
			uniquePixels = config.N_PIXELS
			createFilters()
			"""Visualization effect to display on the LED strip"""
			# Initialize LEDs
			led.update()
			# Start listening to live audio stream
			start_stream(microphone_update,6)
		elif cycleCount==7:
			pixels.fill((255,0,0))
			pixels.show()
		elif cycleCount==8:
			pixels.fill((0,0,255))
			pixels.show()
		elif cycleCount==9:
			pixels.fill((255,0,255))
			pixels.show()
	signal.signal(signal.SIGINT, signal_handler)
	signal.pause()
