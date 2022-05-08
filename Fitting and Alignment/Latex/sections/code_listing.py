# Randy Bullock Fit
def randy_bullock_fit(points):
    N, X, Y = N = len(points), points[:, 0], points[:, 1]
    x_bar, y_bar = np.mean(X), np.mean(Y)
    U, V = X - x_bar, Y - y_bar
    s_uu, s_uv, s_vv = np.sum(U * U), np.sum(U * V), np.sum(V * V)
    s_uuu, s_uvv, s_vuu, s_vvv = np.sum(
        U * U * U), np.sum(U * V * V), np.sum(V * U * U), np.sum(V * V * V)
    A = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    B = np.array([[1 / 2 * (s_uuu + s_uvv)], [1 / 2 * (s_vvv + s_vuu)]])
    [[u_c], [v_c]] = np.linalg.inv(A) @ B
    alpha = math.pow(u_c, 2) + math.pow(v_c, 2) + (s_uu + s_vv) / N
    r, c_x, c_y = sqrt(alpha), u_c + x_bar, v_c + y_bar
    circ = Circle(c_x, c_y, r)
    return circ


# Calculation of Homography
def calcHomography(point1,point2):
    for point1, point2 in zip(p1, p2):
        A[row, 3:6] = -point2[2] * point1
        A[row, 6:9] = point2[1] * point1
        A[row + 1, 0:3] = point2[2] * point1
        A[row + 1, 6:9] = -point2[0] * point1
        row += 2
    # Singular Value decomposition of A
    U, D, VT = np.linalg.svd(A)
    h = VT[-1].reshape(3,3)
    return Homography(H)


# RANSAC Algorithm
 def fit_model(data, model, threshold=0,min_iteration=0):
    num_iterations,iterations_done = math.inf,0
    max_inlier_count,best_model = 0,None
    desired_prob,data_size = 0.95,len(data)

    # Minimum number of sample requirement (circle -> 3 , Homography -> 5)
    num_sample= 3 if model=='circle'  else 5

    while num_iterations > iterations_done:
        np.random.shuffle(data)
        sample_points = data[:num_sample]
        estimated_model = generate_model_from_points(sample_points)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_model = estimated_model
            prob_outlier = 1 - inlier_count / data_size
            num_iterations =  log(1 - desired_prob) /log(1 - (1 - prob_outlier) ** num_sample)
            
            # Adjusting maximum iteration requirement according to maximum inlier count
            if min_iteration>0 and num_iterations < min_iteration:
                num_iterations = min_iteration
        iterations_done = iterations_done + 1
    final_model = generate_model_from_points(inliers)
    return final_model


# Extracting Co-ordinates of matching points using SIFT Operator
def points_extractor(image1,image2):
    sift = cv.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2, k=2)

    # Apply ratio test
    good = []

    for m,n in matches:
        if m.distance < .7*n.distance:
            good.append([m])

    matching_points = [[keypoints_1[mat[0].queryIdx].pt, keypoints_2[mat[0].trainIdx].pt]  for mat in good ]
    return matching_points

h_1_5=Homography(h_4_5.H@h_3_4.H@h_2_3.H@h_1_2.H)