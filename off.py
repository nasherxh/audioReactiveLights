import board
import neopixel
from time import sleep
pixels= neopixel.NeoPixel(board.D18, 100)

def LEDsOff():
	pixels.fill((0,0,0))
	pixels.show()
