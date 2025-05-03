from dolfin import *
import numpy as np

N = 10000

# 创建网格和函数空间
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# 定义边界条件
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)

# 准备存储解的数组 (100, 32, 32)
solutions_array = np.zeros((2, N, 256, 256))

# 生成32x32的均匀中点坐标
n_points = 256
x_coords = np.linspace(1/(2*n_points), 1-1/(2*n_points), n_points)
y_coords = np.linspace(1/(2*n_points), 1-1/(2*n_points), n_points)
points = np.array([[x, y] for y in y_coords for x in x_coords])

# 定义试验和测试函数
u = TrialFunction(V)
v = TestFunction(V)

for i in range(N):
    print(f"Processing solution {i+1}/{N}")
    
    # 随机生成源项f (不同问题可以修改这部分)
    A = np.random.uniform(1, 10)
    kx = np.random.randint(1, 5)
    ky = np.random.randint(1, 5)
    f = Expression('A*sin(kx*x[0])*cos(ky*x[1])', 
                 degree=2, 
                 A=A, kx=kx, ky=ky)
    
    # 定义并求解泊松方程
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    u_sol = Function(V)
    solve(a == L, u_sol, bc)
    
    # 在32x32中点上采样解
    u_values = np.array([u_sol(p) for p in points])
    f_values = np.array([f(p) for p in points])
    solutions_array[0][i] = u_values.reshape((n_points, n_points))
    solutions_array[1][i] = f_values.reshape((n_points, n_points))

# 验证输出数组形状
print("Final solutions array shape:", solutions_array.shape)  # 应为 (100, 32, 32)

# 保存为.npy文件
np.save('poisson_solutions_100x32x32.npy', solutions_array)

# 示例：可视化第0个解
import matplotlib.pyplot as plt
# plt.imshow(solutions_array[0], origin='lower', extent=[0,1,0,1])
plt.imshow(solutions_array[0][0], origin='lower', extent=[0,1,0,1])
plt.colorbar()
plt.title("First Solution (of {N})")
plt.savefig("first_solution.pdf")