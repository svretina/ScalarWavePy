from scalarwavepy import analytic

CFL = 0.2
AMPLITUDE = 8
SIGMA = 1 / 400
CENTER = 0.4
PULSE = analytic.Gaussian(CENTER, AMPLITUDE, SIGMA)
