import matplotlib.pyplot as plt

from physics import simulate
from visualization import animate

if __name__ == "__main__":
    t, s, phi = simulate()
    animation = animate(t, s, phi)
    plt.show()
