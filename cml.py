import numpy as np
import mido
from PyQt5 import QtWidgets
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time

class CoupledMapLattice:
    def __init__(self, size, coupling, map_function='logistic', map_params=None, boundary='periodic', initial_condition='random'):
        self.size = size
        self.map_params = map_params
        self.coupling = coupling
        self.set_map_function(map_function, map_params)
        self.set_boundary_condition(boundary)
        self.set_initial_conditions(initial_condition)

    def set_map_function(self, map_function, map_params=None):
        self.map_params = map_params
        if map_function == 'linear':
            slope = self.map_params.get('slope', 1.0)
            intercept = self.map_params.get('intercept', 0.0)
            self.f = lambda x: np.clip(slope * x + intercept, -1, 1)
        elif map_function == 'logistic':
            r = self.map_params.get('r', 3.9)
            self.f = lambda x: np.clip(r * x * (1 - x), -1, 1)
        elif map_function == 'circular':
            omega = self.map_params.get('omega', 0.5)
            k = self.map_params.get('k', 1.0)
            self.f = lambda x: np.clip((x + omega - k / (2 * np.pi) * np.sin(2 * np.pi * x)) % 1, -1, 1)
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

    def set_initial_conditions(self, initial_condition):
        if isinstance(initial_condition, str) and initial_condition.lower() == 'random':
            self.lattice = np.random.random(self.size)
        elif isinstance(initial_condition, (int, float)):
            self.lattice = np.full(self.size, initial_condition)
        elif isinstance(initial_condition, (list, np.ndarray)) and len(initial_condition) == self.size:
            self.lattice = np.array(initial_condition)
        else:
            raise ValueError("Invalid initial condition. Use 'random', a constant value, or a vector of the correct size.")

    def step(self):
        try:
            fx = self.f(self.lattice)
            left, right = self.boundary(fx)
            self.lattice = np.clip((1 - self.coupling) * fx + self.coupling / 2 * (left + right), 0, 1)
        except Exception as e:
            print(f"Error in step: {e}")
        return self.lattice

    def run(self, steps):
        return np.array([self.step() for _ in range(steps)])

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Declare a new CML
        self.cml = CoupledMapLattice(
            size=100, 
            coupling=0.1, 
            map_function='logistic', 
            map_params={'r': 3.91}, 
            # map_function='linear', 
            # map_params={'slope': 1.3, 'intercept': 0.1}, 
            boundary='periodic',
            initial_condition='random'  # or use a constant value or a list of floats
        )
        
        # Timer for rate-limiting updates
        self.last_update_time = 0
        self.update_interval = 1 / 30  # 30 FPS

        self.initUI()
        # self.startMIDI()

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
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            try:
                self.cml.set_initial_conditions("random")
                lattice_evolution = self.cml.run(1000)
                self.figure.clear()
                
                # # Calculate the median value of the lattice
                # median_value = np.median(lattice_evolution)
                min_value = np.min(lattice_evolution)
                max_value = np.max(lattice_evolution)
                
                # Use a diverging colormap with the midpoint set to the median
                plt.imshow(lattice_evolution, aspect='auto', cmap='coolwarm', vmin=min_value, vmax=max_value)
                plt.colorbar()
                plt.title('Coupled Map Lattice Evolution')
                plt.xlabel('Lattice Site')
                plt.ylabel('Time Step')
                plt.clim(min_value, max_value)  # Set color limits to min and max values
                plt.gca().set_clim(min_value, max_value)  # Ensure color limits are set for the diverging colormap
                
                self.canvas.draw()
                self.last_update_time = current_time
            except Exception as e:
                print(f"Error updating plot: {e}")

    def startMIDI(self):
        self.midi_input = mido.open_input('16n Port 1')
        self.midi_input.callback = self.midi_callback

    def midi_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
          if msg.type == 'control_change':
            if msg.control > 31 and msg.control < 35:
              if msg.control == 32:
                  self.cml.coupling = msg.value / 127.0
              elif msg.control == 33:
                  self.cml.set_map_function(
                      'linear',
                      {
                          'slope': msg.value / 127.0 * 1 - 0.5,
                          'intercept': self.map_params.get('intercept', 1.0) / 127.0 * 1 - 0.5,
                      }
                  )
                  print(f"cc {msg.control} {msg.value}")
              elif msg.control == 34:
                  self.cml.set_map_function(
                      'linear',
                      {
                          'slope': self.map_params.get('slope', 1.0) / 127.0 * 4,
                          'intercept': msg.value / 127.0 * 4,
                      }
                  )
                  print(f"cc {msg.control} {msg.value}")
              self.update_plot()
              self.last_update_time = current_time

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
