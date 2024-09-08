import numpy as np
import mido
from PyQt5 import QtWidgets
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time

class CoupledMapLattice:
    def __init__(self, size, coupling, map_function='logistic', map_params=None, boundary='periodic'):
        self.size = size
        self.map_params = map_params
        self.coupling = coupling
        self.set_map_function(map_function, map_params)
        self.set_boundary_condition(boundary)
        self.lattice = np.random.random(size)

    def set_map_function(self, map_function, map_params=None):
        self.map_params = map_params
        if map_function == 'linear':
            slope = map_params.get('slope', 1.0)
            intercept = map_params.get('intercept', 0.0)
            self.f = lambda x: slope * x + intercept
        elif map_function == 'logistic':
            r = map_params.get('r', 3.9)
            self.f = lambda x: r * x * (1 - x)
        elif map_function == 'circular':
            omega = map_params.get('omega', 0.5)
            k = map_params.get('k', 1.0)
            self.f = lambda x: (x + omega - k / (2 * np.pi) * np.sin(2 * np.pi * x)) % 1
        else:
            raise ValueError("Unsupported map function")

    def set_boundary_condition(self, boundary, value=None):
        if boundary == 'periodic':
            self.boundary = lambda x: (np.roll(x, 1), np.roll(x, -1))
        elif boundary == 'antiperiodic':
            self.boundary = lambda x: (-np.roll(x, 1), -np.roll(x, -1))
        elif boundary == 'fixed':
            if value is None:
                self.boundary = lambda x: (np.pad(x[1:-1], (1, 1), 'edge'), np.pad(x[1:-1], (1, 1), 'edge'))
            elif isinstance(value, (int, float)):
                self.boundary = lambda x: (np.pad(x[1:-1], (1, 1), 'constant', constant_values=value),
                                            np.pad(x[1:-1], (1, 1), 'constant', constant_values=value))
            else:
                raise ValueError("Invalid fixed boundary value")
        else:
            raise ValueError("Unsupported boundary condition")

    def step(self):
        fx = self.f(self.lattice)
        left, right = self.boundary(fx)
        self.lattice = (1 - self.coupling) * fx + self.coupling / 2 * (left + right)
        return self.lattice

    def run(self, steps):
        return np.array([self.step() for _ in range(steps)])

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.cml = CoupledMapLattice(size=100, coupling=0.1, map_function='linear', map_params={'slope': 1.3, 'intercept': 0.1}, boundary='periodic')
        self.initUI()
        self.startMIDI()
        
        # Timer for rate-limiting updates
        self.last_update_time = 0
        self.update_interval = 1 / 30  # 30 FPS

    def initUI(self):
        self.setWindowTitle('Coupled Map Lattice Control')
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.update_plot()
        self.show()

    def update_plot(self):
        self.figure.clear()
        plt.imshow(self.cml.run(100), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Coupled Map Lattice Evolution')
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        self.canvas.draw()

    def startMIDI(self):
        self.midi_input = mido.open_input('16n Port 1')  # Replace with your MIDI port name
        self.midi_input.callback = self.midi_callback

    def midi_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
          if msg.type == 'control_change':
            if msg.control == 32:
                self.cml.coupling = msg.value / 127.0
            elif msg.control == 33:
                self.cml.set_map_function(
                    'linear',
                    {
                        'slope': msg.value / 127.0 * 4,
                        'intercept': self.map_params.get('intercept', 1.0) / 127.0 * 4,
                    }
                )
            elif msg.control == 34:
                self.cml.set_map_function(
                    'linear',
                    {
                        'slope': self.map_params.get('slope', 1.0) / 127.0 * 4,
                        'intercept': msg.value / 127.0 * 4,
                    }
                )
            self.update_plot()  # Call update_plot to check for rate-limiting
            self.last_update_time = current_time  # Update the last update time

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
