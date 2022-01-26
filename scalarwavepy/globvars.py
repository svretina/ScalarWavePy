from scalarwavepy import analytic

CFL = 0.4
AMPLITUDE = 8
SIGMA = 1 / 400
CENTER = -0.2
PULSE = analytic.Gaussian(CENTER, AMPLITUDE, SIGMA)
