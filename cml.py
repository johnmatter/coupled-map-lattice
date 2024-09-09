import numpy as np
import mido
from PyQt5 import QtCore, QtWidgets
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
            r = self.map_params.get('r', 6.052)
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
            self.map_params = {'r': random.uniform(0, 10)}
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

class MultiRowComboBox(QtWidgets.QComboBox):
    def __init__(self, items, max_rows=10, parent=None):
        super().__init__(parent)
        self.setView(QtWidgets.QListView())
        self.view().setWrapping(True)
        self.view().setViewMode(QtWidgets.QListView.ListMode)
        self.view().setFlow(QtWidgets.QListView.TopToBottom)
        self.addItems(items)
        
        item_width = self.view().sizeHintForColumn(0)
        self.view().setMinimumWidth(item_width * 6)
        
        row_height = self.view().sizeHintForRow(0)
        self.view().setMaximumHeight(row_height * max_rows)
        self.view().setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        # Set a fixed height for the combo box itself
        self.setFixedHeight(row_height * 2)  # Adjust as needed

    def showPopup(self):
        super().showPopup()
        # Adjust the size of the popup
        self.view().setGridSize(QtCore.QSize(self.view().sizeHintForColumn(0), self.view().sizeHintForRow(0)))
        # Limit the height of the popup
        self.view().setMaximumHeight(self.view().maximumHeight())

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.cml = CoupledMapLattice(
            size=100, 
            coupling=0.1, 
            map_function='logistic', 
            map_params={'r': 6.052}, 
            boundary='periodic',
            initial_condition='random'
        )
        
        self.lattice_evolution = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Coupled Map Lattice Control')
        layout = QtWidgets.QVBoxLayout()

        # Top control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QGridLayout(control_panel)
        control_panel.setFixedHeight(300)

        # Left column
        # Map function selection
        control_layout.addWidget(QtWidgets.QLabel("Map Function:"), 0, 0)
        self.map_function_combo = QtWidgets.QComboBox()
        self.map_function_combo.addItems(['logistic', 'linear', 'circular'])
        self.map_function_combo.currentTextChanged.connect(self.update_map_function)
        control_layout.addWidget(self.map_function_combo, 0, 1)

        # Map function parameters
        self.param_inputs = {}
        self.param_layout = QtWidgets.QGridLayout()
        control_layout.addLayout(self.param_layout, 1, 0, 1, 2)

        # Randomize button
        self.randomize_button = QtWidgets.QPushButton("Randomize Parameters")
        self.randomize_button.clicked.connect(self.randomize_parameters)
        control_layout.addWidget(self.randomize_button, 2, 0, 1, 2)

        # Initial conditions
        control_layout.addWidget(QtWidgets.QLabel("Initial Condition:"), 3, 0)
        self.initial_condition_combo = QtWidgets.QComboBox()
        self.initial_condition_combo.addItems(['random', 'constant', 'custom'])
        self.initial_condition_combo.currentTextChanged.connect(self.update_initial_condition)
        control_layout.addWidget(self.initial_condition_combo, 3, 1)

        self.initial_condition_input = QtWidgets.QLineEdit()
        self.initial_condition_input.setPlaceholderText("Enter value or list")
        control_layout.addWidget(self.initial_condition_input, 4, 0, 1, 2)
        self.initial_condition_input.hide()

        # Right column
        # Lattice size, time steps, etc.
        control_layout.addWidget(QtWidgets.QLabel("Lattice Size:"), 0, 2)
        self.lattice_size_input = QtWidgets.QSpinBox()
        self.lattice_size_input.setRange(10, 1000)
        self.lattice_size_input.setValue(100)
        control_layout.addWidget(self.lattice_size_input, 0, 3)

        control_layout.addWidget(QtWidgets.QLabel("Time Steps:"), 1, 2)
        self.time_steps_input = QtWidgets.QSpinBox()
        self.time_steps_input.setRange(10, 1000)
        self.time_steps_input.setValue(250)
        control_layout.addWidget(self.time_steps_input, 1, 3)

        # Colormap selection
        control_layout.addWidget(QtWidgets.QLabel("Colormap:"), 2, 2)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(plt.colormaps())
        self.cmap_combo.currentTextChanged.connect(self.update_plot)
        control_layout.addWidget(self.cmap_combo, 2, 3)

        # Run button
        self.run_button = QtWidgets.QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        control_layout.addWidget(self.run_button, 3, 2, 1, 2)

        # Add control panel to main layout
        layout.addWidget(control_panel)

        # Plot
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.update_parameter_inputs()
        self.show()

    def update_map_function(self):
        function = self.map_function_combo.currentText()
        self.cml.set_map_function(function, self.cml.initialize_map_params(function))
        self.update_parameter_inputs()

    def update_parameter_inputs(self):
        for i in reversed(range(self.param_layout.count())): 
            self.param_layout.itemAt(i).widget().setParent(None)
        self.param_inputs.clear()

        function = self.map_function_combo.currentText()
        if function == 'logistic':
            self.add_parameter_input('r', self.cml.map_params.get('r', 6.052), 0, 10)
        elif function == 'linear':
            self.add_parameter_input('slope', self.cml.map_params.get('slope', -1.010), -2, 2)
            self.add_parameter_input('intercept', self.cml.map_params.get('intercept', 0.911), -1, 1)
        elif function == 'circular':
            self.add_parameter_input('omega', self.cml.map_params.get('omega', 0.532), 0, 1)
            self.add_parameter_input('k', self.cml.map_params.get('k', 0.845), 0, 2)

    def add_parameter_input(self, name, default, min_val, max_val):
        label = QtWidgets.QLabel(f"{name}:")
        input_box = QtWidgets.QDoubleSpinBox()
        input_box.setDecimals(3)
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

    def randomize_parameters(self):
        current_function = self.map_function_combo.currentText()
        new_params = self.cml.initialize_map_params(current_function)
        self.cml.set_map_function(current_function, new_params)
        
        # Update the Qt fields with the new parameter values
        for name, value in new_params.items():
            if name in self.param_inputs:
                self.param_inputs[name].setValue(value)

    def update_initial_condition(self):
        condition = self.initial_condition_combo.currentText()
        if condition == 'constant' or condition == 'custom':
            self.initial_condition_input.show()
        else:
            self.initial_condition_input.hide()

    def run_simulation(self):
        lattice_size = self.lattice_size_input.value()
        time_steps = self.time_steps_input.value()
        
        self.cml.size = lattice_size
        
        # Set initial conditions
        condition = self.initial_condition_combo.currentText()
        if condition == 'random':
            self.cml.set_initial_conditions('random')
        elif condition == 'constant':
            try:
                value = float(self.initial_condition_input.text())
                self.cml.set_initial_conditions(value)
            except ValueError:
                print("Invalid constant value. Using random initial conditions.")
                self.cml.set_initial_conditions('random')
        elif condition == 'custom':
            try:
                values = [float(x) for x in self.initial_condition_input.text().split(',')]
                if len(values) == lattice_size:
                    self.cml.set_initial_conditions(values)
                else:
                    print("Custom values don't match lattice size. Using random initial conditions.")
                    self.cml.set_initial_conditions('random')
            except ValueError:
                print("Invalid custom values. Using random initial conditions.")
                self.cml.set_initial_conditions('random')
        
        self.lattice_evolution = self.cml.run(time_steps)
        self.update_plot()

    def update_plot(self):
        if self.lattice_evolution is None:
            return

        self.figure.clear()
        
        min_value = np.min(self.lattice_evolution)
        max_value = np.max(self.lattice_evolution)
        
        selected_cmap = self.cmap_combo.currentText()
        plt.imshow(self.lattice_evolution, aspect='auto', cmap=selected_cmap, vmin=min_value, vmax=max_value)
        plt.colorbar()

        # Subtitle with map function and parameters
        map_function = self.map_function_combo.currentText()
        param_str = ', '.join([f"{k}={v:.3f}" for k, v in self.cml.map_params.items()])
        subtitle = f"Map: {map_function.capitalize()}, Parameters: {param_str}"
        
        # Main title
        title = f'CML Evolution\n{subtitle}'
        plt.title(title)
        
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        plt.clim(min_value, max_value)
        
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
