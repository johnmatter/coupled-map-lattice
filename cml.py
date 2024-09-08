import numpy as np
import mido
from PyQt5 import QtWidgets
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import random

class CoupledMapLattice:
    def __init__(self, size, coupling, map_function='logistic', map_params=None, boundary='periodic', initial_condition='random'):
        self.size = size
        self.map_params = map_params if map_params is not None else {}  # Initialize with an empty dict if None
        self.coupling = coupling
        self.set_map_function(map_function, self.map_params)
        self.set_boundary_condition(boundary)
        self.set_initial_conditions(initial_condition)

    def set_map_function(self, map_function, map_params=None):
        # Safety check for map_params initialization
        if map_params is None or not isinstance(map_params, dict):
            self.initialize_map_params(map_function)

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

    def initialize_map_params(self, map_function):
        # Initialize map_params with random values based on the map function
        if map_function == 'logistic':
            self.map_params = {'r': random.uniform(0, 4)}
        elif map_function == 'linear':
            self.map_params = {
                'slope': random.uniform(-2, 2),
                'intercept': random.uniform(-1, 1)
            }
        elif map_function == 'circular':
            self.map_params = {
                'omega': random.uniform(0, 1),
                'k': random.uniform(0, 2)
            }
        else:
            raise ValueError("Unsupported map function for parameter initialization")
        return self.map_params

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
        self.cml = CoupledMapLattice(
            size=100, 
            coupling=0.1, 
            map_function='logistic', 
            map_params={'r': 3.9}, 
            boundary='periodic',
            initial_condition='random'
        )
        
        # Timer for rate-limiting updates
        self.last_update_time = 0
        self.update_interval = 1 / 30  # 30 FPS

        self.initUI()
        # self.startMIDI()

    def initUI(self):
        self.setWindowTitle('Coupled Map Lattice Control')
        layout = QtWidgets.QVBoxLayout()

        # Combined layout for map function and parameters using QGridLayout
        combined_widget = QtWidgets.QWidget()
        combined_layout = QtWidgets.QGridLayout(combined_widget)
        
        # Set fixed height for the combined widget
        combined_widget.setFixedHeight(250)  # Set your desired height here

        # Map function selection
        combined_layout.addWidget(QtWidgets.QLabel("Map Function:"), 0, 0)
        self.map_function_combo = QtWidgets.QComboBox()
        self.map_function_combo.addItems(['logistic', 'linear', 'circular'])
        self.map_function_combo.currentTextChanged.connect(self.update_map_function)
        combined_layout.addWidget(self.map_function_combo, 0, 1)

        # Colormap selection
        combined_layout.addWidget(QtWidgets.QLabel("Colormap:"), 1, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems([
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',  # Sequential
            'coolwarm', 'RdYlBu', 'Spectral', 'PiYG', 'BrBG',    # Divergent
            'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2',       # Qualitative
            'Accent', 'Dark2', 'Paired', 'tab10', 'tab20'       # Miscellaneous
        ])
        self.cmap_combo.currentTextChanged.connect(self.update_plot)  # Update plot on cmap change
        combined_layout.addWidget(self.cmap_combo, 1, 1)

        # Parameter inputs
        self.param_inputs = {}
        self.param_layout = QtWidgets.QGridLayout()
        combined_layout.addLayout(self.param_layout, 2, 0, 1, 2)  # Span across two columns

        # Randomize button
        self.randomize_button = QtWidgets.QPushButton("Randomize Parameters")
        self.randomize_button.clicked.connect(self.randomize_parameters)
        combined_layout.addWidget(self.randomize_button, 3, 0, 1, 2)  # Span across two columns

        # Add combined widget to main layout
        layout.addWidget(combined_widget)

        # Plot
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)  # This will grow to fill the extra space

        self.setLayout(layout)
        self.update_parameter_inputs()
        self.update_plot()
        self.show()

    def update_map_function(self):
        function = self.map_function_combo.currentText()
        
        self.cml.set_map_function(
            function,
            self.cml.initialize_map_params(function)
        )
        self.update_parameter_inputs()
        self.update_plot()

    def update_parameter_inputs(self):
        # Clear existing inputs
        for i in reversed(range(self.param_layout.count())): 
            self.param_layout.itemAt(i).widget().setParent(None)
        self.param_inputs.clear()

        # Add new inputs based on current map function
        function = self.map_function_combo.currentText()
        if function == 'logistic':
            self.add_parameter_input('r', 3.9, 0, 4)
        elif function == 'linear':
            self.add_parameter_input('slope', 1.0, -2, 2)
            self.add_parameter_input('intercept', 0.0, -1, 1)
        elif function == 'circular':
            self.add_parameter_input('omega', 0.5, 0, 1)
            self.add_parameter_input('k', 1.0, 0, 2)

    def add_parameter_input(self, name, default, min_val, max_val):
        label = QtWidgets.QLabel(f"{name}:")
        input_box = QtWidgets.QDoubleSpinBox()
        input_box.setRange(min_val, max_val)
        input_box.setSingleStep(0.001)
        input_box.setValue(default)
        input_box.valueChanged.connect(self.update_parameters)
        
        row = len(self.param_inputs)
        self.param_layout.addWidget(label, row, 0)
        self.param_layout.addWidget(input_box, row, 1)
        self.param_inputs[name] = input_box

    def update_parameters(self):
        params = {name: input_box.value() for name, input_box in self.param_inputs.items()}
        self.cml.set_map_function(self.map_function_combo.currentText(), params)
        self.update_plot()

    def randomize_parameters(self):
        for name, input_box in self.param_inputs.items():
            random_value = random.uniform(input_box.minimum(), input_box.maximum())
            input_box.setValue(random_value)
        self.update_parameters()

    def update_plot(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            try:
                self.cml.set_initial_conditions("random")
                lattice_evolution = self.cml.run(250)
                self.figure.clear()
                
                min_value = np.min(lattice_evolution)
                max_value = np.max(lattice_evolution)
                
                # Use the selected colormap
                selected_cmap = self.cmap_combo.currentText()
                plt.imshow(lattice_evolution, aspect='auto', cmap=selected_cmap, vmin=min_value, vmax=max_value)
                plt.colorbar()
                plt.title('Coupled Map Lattice Evolution')
                plt.xlabel('Lattice Site')
                plt.ylabel('Time Step')
                plt.clim(min_value, max_value)
                
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
                          'intercept': self.map_params.get('intercept', 1.0),
                      }
                  )
                  print(f"cc {msg.control} {msg.value}")
              elif msg.control == 34:
                  self.cml.set_map_function(
                      'linear',
                      {
                          'slope': self.map_params.get('slope', 1.0),
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
