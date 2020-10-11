# audioReactiveLights
audio reactive lights with button to swap functions
Code adapted from: https://github.com/naztronaut/dancyPi-audio-reactive-led

Setup: 
-Lights data cable attached to GPIO pin 18.
-Button attached between GPIO pin 18 and ground. 
-Edit code in 'config' so N_PIXELS is equal to length of LED strip

To use, run function 'main'.
Press button to cycle through different light displays:
rainbowcycle: non audio reactive rainbow which shows entire colour spectrum and rotates it around table
rainbowcycle2: non audio reactive rainbow which cycles through colour spectrum
scroll
scroll2
energy
energy2
spectrum
red
blue
purple
off

