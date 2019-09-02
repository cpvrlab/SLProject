#include "Calibration.h"
#include "tools.h"

void Calibration::init_matrix(Matrix3f &intrinsic_matrix, Vector2f &P, Vector2f &K, float fov, int width, int height)
{ 
    float cx  = width * 0.5f;
    float cy  = height * 0.5f;
    float fy  = cy / tanf(fov * 0.5f * M_PI / 180.0);
    float fx  = fy;

    intrinsic_matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    K << 0, 0;
    P << 0, 0;
}

float Calibration::calibrate_opencv(Matrix3f &intrinsic, Vector2f &P, Vector2f &K, 
                                    int width, int height, 
                                    std::vector<std::vector<cv::Point2f>> &keypoints,
                                    std::vector<std::vector<cv::Point3f>> &worldpoints)
{
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat matrix     = eigen2cv(intrinsic);
    cv::Size size      = cv::Size(width, height);
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << K[0], K[1], P[0], P[1], 0);

    float err = cv::calibrateCamera(worldpoints, keypoints, 
                                    size, matrix, distortion, 
                                    rvecs, tvecs, 
                                    cv::CALIB_USE_INTRINSIC_GUESS | 
                                    cv::CALIB_ZERO_TANGENT_DIST | 
                                    cv::CALIB_FIX_ASPECT_RATIO);

    intrinsic     = cv2eigen (matrix);
    K[0] = distortion.at<double>(0, 0);
    K[1] = distortion.at<double>(1, 0);
    P[0] = distortion.at<double>(2, 0);
    P[1] = distortion.at<double>(3, 0);

    return err;
}

float Calibration::calibrate_opencv(Matrix3f &intrinsic, 
                                    int width, int height, 
                                    std::vector<std::vector<cv::Point2f>> &keypoints,
                                    std::vector<std::vector<cv::Point3f>> &worldpoints)
{
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat matrix     = eigen2cv(intrinsic);
    cv::Size size      = cv::Size(width, height);
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    cv::calibrateCamera(worldpoints, keypoints, 
                        size, matrix, distortion, 
                        rvecs, tvecs, 
                        cv::CALIB_USE_INTRINSIC_GUESS | 
                        cv::CALIB_ZERO_TANGENT_DIST | 
                        cv::CALIB_FIX_ASPECT_RATIO);

    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    float err = reprojectionRMS(matrix, distortion, keypoints, worldpoints, rvecs, tvecs);

    intrinsic     = cv2eigen (matrix);

    return err;
}

float Calibration::calibrate_opencv_fixed_center(Matrix3f &intrinsic, 
                                                int width, int height, 
                                                std::vector<std::vector<cv::Point2f>> &keypoints,
                                                std::vector<std::vector<cv::Point3f>> &worldpoints)
{
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat matrix     = eigen2cv(intrinsic);
    cv::Size size      = cv::Size(width, height);
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    cv::calibrateCamera(worldpoints, keypoints, 
                        size, matrix, distortion, 
                        rvecs, tvecs, 
                        cv::CALIB_USE_INTRINSIC_GUESS | 
                        cv::CALIB_ZERO_TANGENT_DIST | 
                        cv::CALIB_FIX_ASPECT_RATIO);

    matrix.at<double>(0, 2) = width * 0.5;
    matrix.at<double>(1, 2) = height * 0.5;
    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    float err = reprojectionRMS(matrix, distortion, keypoints, worldpoints, rvecs, tvecs);

    intrinsic = cv2eigen (matrix);

    return err;
}

float Calibration::calibrate_brute_force(Matrix3f &intrinsic, Vector2f &P, Vector2f &K, 
                                         int width, int height, 
                                         std::vector<std::vector<cv::Point2f>> &keypoints,
                                         std::vector<std::vector<cv::Point3f>> &worldpoints)
{
    float min_err = 9999999;

    for (int i = 0; i < 20; i++)
    {
        float fov = 30.0 + i;

        Matrix3f matrix;
        Vector2f p;
        Vector2f k;
        init_matrix (matrix, p, k, fov, width, height);
        float err = calibrate_opencv_fixed_center(matrix, width, height, keypoints, worldpoints);

        if (err < min_err)
        {
            float result_fov = get_fovy(matrix);
            if (result_fov > 30 && result_fov < 50)
            {
                min_err = err;
                intrinsic = matrix;
                P = p;
                K = k;
            }
        }
    }

    return min_err;
}

float Calibration::calibrate_unique_view(Matrix3f &intrinsic, Vector2f &P, Vector2f &K, 
                                         int width, int height, 
                                         std::vector<cv::Point2f> &keypoints,
                                         std::vector<cv::Point3f> &worldpoints)
{
    cv::Mat rvec, tvec;
    cv::Mat matrix     = eigen2cv (intrinsic);
    cv::Size size      = cv::Size(width, height);
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << K[0], K[1], P[0], P[1], 0);
    float err = calibrate_unique_view(matrix, distortion, size, rvec, tvec, keypoints, worldpoints);

    intrinsic     = cv2eigen (matrix);
    K[0] = distortion.at<double>(0, 0);
    K[1] = distortion.at<double>(1, 0);
    P[0] = distortion.at<double>(2, 0);
    P[1] = distortion.at<double>(3, 0);

    return err;
}



float Calibration::calibrate_ransac_unique_view(std::vector<Vector2f> &outliers, std::vector<Vector2f> &corrects,
                                                  Matrix3f &intrinsic, Vector2f &P, Vector2f &K, int width, int height, 
                                                  float &total_error, int nb_iter, int percent_correct, float threshold, 
                                                  int selection_percent, 
                                                  std::vector<cv::Point2f> keypoints, std::vector<cv::Point3f> worldpoints)
{

    cv::Size size      = cv::Size(width, height);
    //Best Parameters Backup
    cv::Mat ini_matrix     = eigen2cv (intrinsic);
    cv::Mat ini_distortion = (cv::Mat_<float>(5, 1) << K[0], K[1], P[0], P[1], 0);

    //RANSAC
    bool has_solution = false;
    int d = (percent_correct / 100.0) * worldpoints.size();

    //Create temporary vectors
    std::vector<bool> subst;
    std::vector<cv::Point3f> p3d_sub1;
    std::vector<cv::Point3f> p3d_sub2;
    std::vector<cv::Point2f> p2d_sub1;
    std::vector<cv::Point2f> p2d_sub2;
    std::vector<cv::Point2f> projected;

    for (int k = 0; k < nb_iter || has_solution == false; k++)
    {
        //Clear temporary Vectors
        subst.clear();
        p3d_sub1.clear();
        p3d_sub2.clear();
        p2d_sub1.clear();
        p2d_sub2.clear();
        projected.clear();

        //Temporary model parameters
        cv::Mat matrix = ini_matrix.clone();
        cv::Mat distortion = ini_distortion.clone();

        //Select a given % of all points
        select_random(subst, (selection_percent / 100.0) * worldpoints.size());

        pick_selection(p2d_sub1, p3d_sub1, p2d_sub2, p3d_sub2, subst, keypoints, worldpoints);

        //Get model with subset 1 
        cv::Mat rvec, tvec;

        // Find parameter on selected points
        calibrate_unique_view(matrix, distortion, size, rvec, tvec, p2d_sub1, p3d_sub1);
        // Force distortion to 0 0 0 0 and cx cy to image center
        distortion = ini_distortion.clone();
        matrix.at<double>(0, 2) = size.width * 0.5;
        matrix.at<double>(1, 2) = size.height * 0.5;

        //Normally in ransac, we apply model on complement (subset 2) but here we force distortion = 0
        //and cx cy = center. So we recompute the error over all set
        cv::projectPoints(worldpoints, rvec, tvec, matrix, distortion, projected);
        float curr_total_error = 0;
        int nb_correct = 0;
        p2d_sub2.clear();
        p2d_sub1.clear();

        for (int i = 0; i < worldpoints.size(); i++)
        {
            cv::Point2f d = keypoints[i] - projected[i];
            float t = sqrt(d.x*d.x + d.y*d.y);
            if (t < threshold)
            {
                // Add current point error to total current error
                curr_total_error += t;
                nb_correct++;
                p2d_sub1.push_back(keypoints[i]);
            }
            else
            {
                p2d_sub2.push_back(keypoints[i]);
            }
        }

        // If there is more than d correct elements that fit the current model and the total error with this model
        // is the best we have so far => keeps this model
        if (nb_correct > d && curr_total_error < total_error)
        {
            float fovy = get_fovy(matrix);

            if (fovy > 30 && fovy < 50)
            {
                total_error = curr_total_error;

                intrinsic     = cv2eigen (matrix);
                K[0] = distortion.at<double>(0, 0);
                K[1] = distortion.at<double>(1, 0);
                P[0] = distortion.at<double>(2, 0);
                P[1] = distortion.at<double>(3, 0);

                has_solution = true;

                outliers.clear();
                corrects.clear();

                for (cv::Point2f v : p2d_sub2)
                {
                    outliers.push_back(Vector2f(v.x, v.y));
                }
                for (cv::Point2f v : p2d_sub1)
                {
                    corrects.push_back(Vector2f(v.x, v.y));
                }
            }
        }
    }

    return total_error;
}

bool Calibration::calibrate_ransac(Matrix3f &intrinsic, Vector2f &P, Vector2f &K, int width, int height, float& total_error, 
                                   int nb_iter, int percent_correct, float threshold, int nselect, 
                                   std::vector<std::vector<cv::Point2f>>& keypoints, 
                                   std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    bool has_solution = false;
    int  N            = keypoints.size();
    cv::Size size     = cv::Size(width, height);

    //Create temporary vectors
    std::vector<std::vector<bool>>        selections = gen_binary_vectors(keypoints);
    std::vector<std::vector<cv::Point2f>> skp        = gen_selection_vectors(keypoints);
    std::vector<std::vector<cv::Point3f>> swp        = gen_selection_vectors(worldpoints);
    std::vector<std::vector<cv::Point2f>> nskp       = gen_selection_vectors(keypoints);
    std::vector<std::vector<cv::Point3f>> nswp       = gen_selection_vectors(worldpoints);
    std::vector<cv::Point2f>              projected;
    int                                   d = 0;
    for (int i = 0; i < N; i++) { d += ((percent_correct * keypoints[i].size()) / 100); }

    //Save of initial model parameters
    cv::Mat ini_matrix     = eigen2cv (intrinsic);
    cv::Mat ini_distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    //Model parameters to optimize
    cv::Mat              matrix = ini_matrix;
    std::vector<cv::Mat> rvecs, tvecs;

    for (int k = 0; k < nb_iter; k++)
    {
        //Clear temporary Vectors
        for (int i = 0; i < N; i++)
        {
            skp[i].clear();
            swp[i].clear();
            nskp[i].clear();
            nswp[i].clear();
        }

        //Select nselect of all points
        select_random(selections, nselect);
        pick_selection(skp, swp, nskp, nswp, selections, keypoints, worldpoints);

        // Compute parameters for nselect points per frame
        calibrate_brute_force(matrix, size, rvecs, tvecs, skp, swp);

        cout << "fov = " <<  get_fovy(matrix) << endl;

        float curr_total_error = 0;
        int nb_correct = 0;

        for (int i = 0; i < N; i++)
        {
            projected.clear();
            cv::projectPoints(worldpoints[i], rvecs[i], tvecs[i], matrix, ini_distortion, projected);

            for (int j = 0; j < worldpoints[i].size(); j++)
            {
                cv::Point2f p = keypoints[i][j] - projected[j];
                float       t = sqrt(p.x * p.x + p.y * p.y);
                if (t < threshold)
                {
                    // Add current point error to total current error
                    curr_total_error += t;
                    nb_correct++;
                }
            }
        }

        // If there is more than d elements that fit the current model and
        //  the total error with this model is the lowest we have so far => keeps this model
        if (nb_correct >= d && curr_total_error < total_error)
        {
           float fov = get_fovy(matrix); 

            if (fov > 30 && fov < 50)
            {
                total_error  = curr_total_error;
                intrinsic    = cv2eigen (matrix);

                has_solution = true;
            }
        }
    }
    return has_solution;
};


// Private


float Calibration::reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
                                   std::vector<std::vector<cv::Point2f>>& vvP2D,
                                   std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                   const std::vector<cv::Mat>& rvecs,
                                   const std::vector<cv::Mat>& tvecs)
{
    std::vector <cv::Point2f> projected;
    float error = 0;
    int n = 0;
    for (int i = 0; i < vvP3Dw.size(); i++)
    {
        cv::projectPoints(vvP3Dw[i], rvecs[i], tvecs[i], intrinsic, distortion, projected);

        error += cv::norm(vvP2D[i], projected, cv::NORM_L2);
        n = n + vvP2D[i].size();
    }
    return error / n;
}


float Calibration::calibrate_unique_view(cv::Mat &matrix, cv::Mat &distortion, 
                                         cv::Size &size, 
                                         cv::Mat &rvec, cv::Mat &tvec, 
                                         std::vector<cv::Point2f> &keypoints,
                                         std::vector<cv::Point3f> &worldpoints)
{
    std::vector<std::vector<cv::Point3f>> points3ds(1, worldpoints);
    std::vector<std::vector<cv::Point2f>> points2ds(1, keypoints);
    std::vector<cv::Mat>                  rvecs(1, rvec);
    std::vector<cv::Mat>                  tvecs(1, tvec);

    float err = cv::calibrateCamera(points3ds, points2ds, 
            size, matrix, distortion, 
            rvecs, tvecs, 
            cv::CALIB_USE_INTRINSIC_GUESS | 
            cv::CALIB_ZERO_TANGENT_DIST | 
            cv::CALIB_FIX_ASPECT_RATIO);
    rvec = rvecs[0];
    tvec = tvecs[0];
    return err;
}

float Calibration::calibrate_opencv_fixed_center(cv::Mat &matrix, 
                                                 cv::Size &size, 
                                                 std::vector<cv::Mat> &rvecs, 
                                                 std::vector<cv::Mat> &tvecs,
                                                 std::vector<std::vector<cv::Point2f>> &keypoints,
                                                 std::vector<std::vector<cv::Point3f>> &worldpoints)
{
   
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    cv::calibrateCamera(worldpoints, keypoints, 
                        size, matrix, distortion, 
                        rvecs, tvecs, 
                        cv::CALIB_USE_INTRINSIC_GUESS | 
                        cv::CALIB_ZERO_TANGENT_DIST | 
                        cv::CALIB_FIX_ASPECT_RATIO);

    matrix.at<double>(0, 2) = size.width * 0.5;
    matrix.at<double>(1, 2) = size.height * 0.5;
    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    return reprojectionRMS(matrix, distortion, keypoints, worldpoints, rvecs, tvecs);
}

float Calibration::calibrate_brute_force(cv::Mat &intrinsic, 
                                         cv::Size &size, 
                                         std::vector<cv::Mat> &rvecs, 
                                         std::vector<cv::Mat> &tvecs,
                                         std::vector<std::vector<cv::Point2f>> &keypoints,
                                         std::vector<std::vector<cv::Point3f>> &worldpoints)
{
    float min_err = 9999999;
    cv::Mat matrix;

    for (int i = 0; i < 10; i++)
    {
        float fov = 30.0 + 2 * i;
        std::vector<cv::Mat> _rvecs;
        std::vector<cv::Mat> _tvecs;

        init_matrix (matrix, fov, size);
        float err = calibrate_opencv_fixed_center(matrix, size, _rvecs, _tvecs, keypoints, worldpoints);

        if (err < min_err)
        {
            float result_fov = get_fovy(matrix);
            if (result_fov > 30 && result_fov < 50)
            {
                min_err = err;
                intrinsic = matrix.clone();
                rvecs = _rvecs;
                tvecs = _tvecs;
            }
        }
    }

    return min_err;
}

void Calibration::init_matrix(cv::Mat &mat, float fov, cv::Size &size)
{
    float cx  = (float)size.width * 0.5f;
    float cy  = (float)size.height * 0.5f;
    float fy  = cy / tanf(fov * 0.5f * M_PI / 180.0);
    float fx  = fy;
    mat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}


