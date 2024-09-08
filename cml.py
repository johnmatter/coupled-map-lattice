import numpy as np

class CoupledMapLattice:
    def __init__(self, size, coupling, map_function='logistic', map_params=None, boundary='periodic'):
        self.size = size
        self.coupling = coupling
        self.set_map_function(map_function, map_params)
        self.set_boundary_condition(boundary)
        self.lattice = np.random.random(size)

    def set_map_function(self, map_function, map_params=None):
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

# Example usage
cml = CoupledMapLattice(size=100, coupling=0.3, map_function='logistic', map_params={'r': 3.32}, boundary='periodic')
result = cml.run(steps=4000)

# Plot the result
import matplotlib.pyplot as plt
plt.imshow(result, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Coupled Map Lattice Evolution')
plt.xlabel('Lattice Site')
plt.ylabel('Time Step')
plt.show()
