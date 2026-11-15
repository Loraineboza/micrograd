# Micrograd numerical gradient check

Команда: python examples/gradcheck.py

Среда GitHub Actions:
- Python 3.11.18
- NumPy 2.4.6
- dtype: float64
- central finite-difference step: 1.0e-06
- tolerance: 9.693318610612e-05

Результат:
- max_abs_error_W: 2.218909500584e-10
- max_abs_error_b: 5.379396927907e-11
- status: Gradient check PASSED

CI run: https://github.com/Loraineboza/micrograd/actions/runs/43743622569

Аналитические и численные градиенты:

W analytic = numeric:
[[ 0.71579675,  0.50021337],
 [-0.36904855, -0.35354785],
 [ 0.35664533,  0.61813603]]

b analytic = numeric:
[0.57004469, -0.95933186]
