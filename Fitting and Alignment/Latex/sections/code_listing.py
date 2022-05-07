# Q2
# Function to create overlay
def overlay_image(im_src, im_dst, pts_src, pts_dst):
    h = cv.getPerspectiveTransform(pts_src, pts_dst)
    transformed_image = cv.warpPerspective(
        im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    b = (transformed_image[:, :, 0] == 0)*im_dst[:, :, 0]
    g = (transformed_image[:, :, 1] == 0)*im_dst[:, :, 1]
    r = (transformed_image[:, :, 2] == 0)*im_dst[:, :, 2]
    masked = np.dstack((b, g, r))
    overlayed = cv.add(masked, transformed_image)
    return overlayed


# Randy Bullock Fit
def randy_bullock_fit(points):
    N = len(points)
    X, Y = points[:, 0], points[:, 1]
    x_bar, y_bar = np.mean(X), np.mean(Y)
    U, V = X - x_bar, Y - y_bar
    s_uu, s_uv, s_vv = np.sum(U * U), np.sum(U * V), np.sum(V * V)
    s_uuu, s_uvv, s_vuu, s_vvv = np.sum(
        U * U * U), np.sum(U * V * V), np.sum(V * U * U), np.sum(V * V * V)
    A = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    B = np.array([[1 / 2 * (s_uuu + s_uvv)], [1 / 2 * (s_vvv + s_vuu)]])
    [[u_c], [v_c]] = np.linalg.inv(A) @ B
    alpha = math.pow(u_c, 2) + math.pow(v_c, 2) + (s_uu + s_vv) / N
    r = math.sqrt(alpha)
    c_x, c_y = u_c + x_bar, v_c + y_bar
    circ = Circle(c_x, c_y, r)
    return circ


# Calculation of Homography
def calcHomography(m_points):
    """
    The normalized DLT for 2D homographs.
    """
    p1, p2 = m_points[:, 0], m_points[:, 1]
    # Normalizing the points using predefined function
    T1, p1 = Homography.normalizePoints(p1)
    T2, p2 = Homography.normalizePoints(p2)
    # Initialising an array to keep the coefficient matrix
    A = np.zeros((2 * len(p1), 9))
    row = 0
    # Filling rows of the matrix according to the expressions
    for point1, point2 in zip(p1, p2):
        A[row, 3:6] = -point2[2] * point1
        A[row, 6:9] = point2[1] * point1
        A[row + 1, 0:3] = point2[2] * point1
        A[row + 1, 6:9] = -point2[0] * point1
        row += 2
    # Singular Value decomposition of A
    U, D, VT = np.linalg.svd(A)
    h = VT[-1]
    # Reshaping to get 3x3 homography
    H = h.reshape((3, 3))
    # Renormalization
    H = np.linalg.inv(T2).dot(H).dot(T1)
    return Homography(H)


# RANSAC Algorithm
 def fit_model(data, model, threshold=0,min_iteration=0):
        """
        Fitting Model Using RANSAC Algorithm
        """
        num_iterations,iterations_done = math.inf,0
        max_inlier_count,best_model,model_points = 0,None,None
        desired_prob,data_size = 0.95,len(data)

        # Assigning model parameters
        num_sample= 3 if model=='circle'  else 5

        while num_iterations > iterations_done:
            np.random.shuffle(data)
            sample_points = data[:num_sample]
            estimated_model = generate_model_from_points(sample_points)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model
                model_points = sample_points

                prob_outlier = 1 - inlier_count / data_size
                try:
                    num_iterations =  math.log(1 - desired_prob) /
                         math.log(1 - (1 - prob_outlier) ** num_sample)
                    if min_iteration>0 and num_iterations < min_iteration:
                        num_iterations = min_iteration
                except:
                    pass
            iterations_done = iterations_done + 1

        return best_model, model_points, iterations_done, max_inlier_count


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