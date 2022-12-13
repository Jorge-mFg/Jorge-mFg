import math


# Factorial methods: for loop serial product; recursive; math.factorial(\integer)
def _factorial(n):
    return 1 if n == 0 else n * _factorial(n - 1)
def factorial(n):
    prod = 1
    for num in range(2, n + 1):
        prod = prod * num
    return prod
def factorial_(n):
    return math.factorial(n)

# Absolute value or modulus function of a number
def modulus(x):
    return math.sqrt(x ** 2)
def modulus_(x):
    return abs(x)

# Sign function
def sign(x):
    return math.sqrt(x ** 2) / x
def sign_(x):
    return abs(x) / x

# Heaviside function
def heaviside(x, x0):
    return (abs(x - x0) / (x - x0) + 1) / 2
def heaviside_(x, x0):
    return 1 * (x > x0)

# Delta-Dirac function
def dirac_(x, x0):
    return 1 * (x == x0)
def dirac(x, x0, d):
    return abs(x - x0) / (x - x0) - abs(x - x0 - d) / (x - x0 - d)

# Rotation of a list
def _rotate(a, r):
    r = r % len(a)
    return a[-r:] + a[:-r]
def rotate(a, r):
    for i in range(r):
        a.append(a.pop(0))
    return a

# Number of Unordered Combinations
def c(n, k):
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


x = range(0, 4)
y = range(0, 6)
x_plus_y = []
k, i = 0, 0
Win = False
while Win == False:
    if i % 6 == 0:
        k = k + 1
        i = 0
    if k > len(x):
        break
    if (x[k - 1] + y[i]) == 3:
        print(x[k - 1], y[i])
        Win = True
    # print(k, i)
    # print(x[k], y[i])
    x_plus_y.append(x[k - 1] + y[i])
    # print(x_plus_y)
    i = i + 1

print(x_plus_y)
print(len(x_plus_y))


# Calculate partial derivatives.
