# part one

# Randy Bullock Fit
def randy_bullock_fit(points):
    N = len(points)
    X, Y = points[:, 0], points[:, 1]
    x_bar, y_bar = np.mean(X), np.mean(Y)
    U, V = X - x_bar, Y - y_bar
    s_uu, s_uv, s_vv = np.sum(U * U), np.sum(U * V), np.sum(V * V)
    s_uuu, s_uvv, s_vuu, s_vvv = np.sum(U * U * U), np.sum(U * V * V), np.sum(V * U * U), np.sum(V * V * V)
    A = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    B = np.array([[1 / 2 * (s_uuu + s_uvv)], [1 / 2 * (s_vvv + s_vuu)]])
    [[u_c], [v_c]] = np.linalg.inv(A) @ B
    alpha = math.pow(u_c, 2) + math.pow(v_c, 2) + (s_uu + s_vv) / N
    r = math.sqrt(alpha)
    c_x, c_y = u_c + x_bar, v_c + y_bar
    circ = Circle(c_x, c_y, r)
    return circ