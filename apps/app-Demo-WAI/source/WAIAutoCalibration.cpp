#include <WAIAutoCalibration.h>
#include <WAICalibration.h>
#include <CVCapture.h>
using namespace cv;
#define NB_SAMPLES 10

AutoCalibration::AutoCalibration(int width, int height)
{
    _imgSize         = cv::Size(width, height);
    reset();
}

void AutoCalibration::setCameraParameters(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2)
{
    _cameraMat  = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    _distortion = (Mat_<double>(5, 1) << k1, k2, p1, p2, 0);
    _cameraFovDeg = calcCameraFOV();
}

void AutoCalibration::reset()
{
    WAICalibration::reset();
    _vvP2D.clear();
    _vvP3Dw.clear();
    _error = 9999.0;
}


bool AutoCalibration::tryCalibrateRansac(std::vector<cv::Point2f> vP2D, std::vector<cv::Point3f> vP3Dw)
{
    if (vP3Dw.size() < 30)
        return false;

    // Add new matches to the list if they are different than the old one
    if (_vvP3Dw.size() == 0)
    {
        _vvP3Dw.push_back(vP3Dw);
        _vvP2D.push_back(vP2D);
        cout << "add view" << endl;
    }
    else
    {
        Point3f min, max;
        Point3f m_new;
        Point3f m_old;
        float   dist;

        mean_position(m_old, max, min, _vvP3Dw.back());
        dist = norm(max - min) * 0.5;
        mean_position(m_new, max, min, vP3Dw);
        dist += norm(max - min) * 0.5;

        if (norm(m_new - m_old) > 0.2 * dist)
        {
           _vvP3Dw.push_back(vP3Dw);
           _vvP2D.push_back(vP2D);
            cout << "add view" << endl;
        }
    }

    if (_vvP2D.size() < NB_SAMPLES)
        return false;

    cout << "RANSAC on " << NB_SAMPLES << " frames with 50 iter" << endl;
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
    cv::Mat intrinsic;
    float error = 99999;

    if (calibrate_ransac(intrinsic, distortion, _imgSize, error, 50, 35, 5, 4, _vvP2D, _vvP3Dw))
    {
        float fov = calcCameraFOV(intrinsic);
        cout << " fov = " << fov << " min_err = " << _error << endl;
        cout << "= camera matrix =" << endl << intrinsic << endl;
    }
    _vvP2D.clear();
    _vvP3Dw.clear();

    if (error < _error)
    {
        _intrinsic = intrinsic.clone();
        return true;
    }
    return false;
}

bool AutoCalibration::tryCalibrateBruteForce(std::vector<cv::Point2f> vP2D, std::vector<cv::Point3f> vP3Dw)
{
    if (vP3Dw.size() < 30)
        return false;

    // Add new matches to the list if they are different than the old one
    if (_vvP3Dw.size() == 0)
    {
        _vvP3Dw.push_back(vP3Dw);
        _vvP2D.push_back(vP2D);
    }
    else
    {
        Point3f min, max;
        Point3f m_new;
        Point3f m_old;
        float   dist;

        mean_position(m_old, max, min, _vvP3Dw.back());
        dist = norm(max - min) * 0.5;
        mean_position(m_new, max, min, vP3Dw);
        dist += norm(max - min) * 0.5;

        if (norm(m_new - m_old) > 0.2 * dist)
        {
           _vvP3Dw.push_back(vP3Dw);
           _vvP2D.push_back(vP2D);
        }
    }

    if (_vvP2D.size() < NB_SAMPLES)
        return false;

    savePoints();

    cv::Mat intrinsic;
    std::vector<cv::Mat> rvecs, tvecs;

    float err;
    if(!calibrateBruteForce(intrinsic, _vvP2D, _vvP3Dw, rvecs, tvecs, err))
    {
        _vvP2D.clear();
        _vvP3Dw.clear();
        return false;
    }

    _vvP2D.clear();
    _vvP3Dw.clear();

    if (err < _error)
    {
        _intrinsic = intrinsic.clone();
        return true;
    }

    return false;
}

bool AutoCalibration::calibrateBruteForce(cv::Mat &intrinsic,
                                             std::vector<std::vector<cv::Point2f>>& vvP2D,
                                             std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                             std::vector<cv::Mat>& rvecs,
                                             std::vector<cv::Mat>& tvecs,
                                             float &error)

{
    error = 999999999.0;

    bool has_solution = false;
    computeMatrix(intrinsic, 40.0);

    for (int i = 0; i < 20; i++)
    {
        float fov = 30 + i;

        std::vector<cv::Mat> rotvecs, trvecs;
        cv::Mat matrix;
        computeMatrix(matrix, fov);

        float err = calibrate_opencv_no_distortion_fixed_center (matrix, _imgSize, rotvecs, trvecs, vvP2D, vvP3Dw);

        if (err < error)
        {
            float rfov = calcCameraFOV(matrix);

            if (rfov > 30 && rfov < 50)
            {
                error     = err;
                intrinsic = matrix.clone();
                rvecs     = rotvecs;
                tvecs     = trvecs;
                has_solution = true;
            }
        }
    }

    return has_solution;
}

void AutoCalibration::mean_position(cv::Point3f& mean, cv::Point3f& max, cv::Point3f& min, std::vector<cv::Point3f>& points3d)
{
    mean = Point3f(0, 0, 0);
    min  = Point3f(9999, 9999, 9999);
    max  = Point3f(-9999, -9999, -9999);

    for (Point3f v : points3d)
    {
        mean += v * (1.0 / points3d.size());
        if (v.x < min.x) { min.x = v.x; }
        if (v.y < min.y) { min.y = v.y; }
        if (v.z < min.z) { min.z = v.z; }

        if (v.x > max.x) { max.x = v.x; }
        if (v.y > max.y) { max.y = v.y; }
        if (v.z > max.z) { max.z = v.z; }
    }
}

float AutoCalibration::calibrate_opencv(cv::Mat& matrix, cv::Mat& distortion, cv::Size& size,
                                        cv::Mat& rvec, cv::Mat& tvec,
                                        std::vector<cv::Point2f>& keypoints,
                                        std::vector<cv::Point3f>& worldpoints)
{
    std::vector<std::vector<cv::Point3f>> points3ds(1, worldpoints);
    std::vector<std::vector<cv::Point2f>> points2ds(1, keypoints);
    std::vector<cv::Mat>                rvecs(1, rvec);
    std::vector<cv::Mat>                tvecs(1, tvec);

    float err = cv::calibrateCamera(points3ds, points2ds, size, matrix, distortion, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO);
    rvec      = rvecs[0];
    tvec      = tvecs[0];
    return err;
}

float AutoCalibration::calibrate_opencv(cv::Mat& matrix, cv::Mat& distortion,
                                        cv::Size& size, std::vector<cv::Mat>& rvecs,
                                        std::vector<cv::Mat>& tvecs,
                                        std::vector<std::vector<cv::Point2f>>& keypoints,
                                        std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    return cv::calibrateCamera(worldpoints, keypoints, size, matrix, distortion, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO);
}

float AutoCalibration::calibrate_opencv_no_distortion(cv::Mat& matrix, cv::Size& size,
                                                      std::vector<cv::Mat>& rvecs,
                                                      std::vector<cv::Mat>& tvecs,
                                                      std::vector<std::vector<cv::Point2f>>& keypoints,
                                                      std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
    cv::calibrateCamera(worldpoints, keypoints, size, matrix, distortion, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO);
    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    return reprojectionRMS(matrix, distortion, keypoints, worldpoints, rvecs, tvecs);
}

float AutoCalibration::calibrate_opencv_no_distortion_fixed_center(cv::Mat& matrix, cv::Size& size,
                                                                   std::vector<cv::Mat>& rvecs,
                                                                   std::vector<cv::Mat>& tvecs,
                                                                   std::vector<std::vector<cv::Point2f>>& keypoints,
                                                                   std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
    cv::calibrateCamera(worldpoints, keypoints, size, matrix, distortion, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO);
    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    matrix.at<double>(0, 2) = size.width * 0.5;
    matrix.at<double>(1, 2) = size.height * 0.5;

    return reprojectionRMS(matrix, distortion, keypoints, worldpoints, rvecs, tvecs);
}

std::vector<std::vector<bool>> gen_binary_vectors(std::vector<std::vector<cv::Point2f>> vvP2D)
{
    std::vector<std::vector<bool>> selections;
    selections.reserve(vvP2D.size());

    for (int i = 0; i < vvP2D.size(); i++)
    {
        std::vector<bool> binary_vector(vvP2D[i].size(), 0);
        selections.push_back(binary_vector);
    }
    return selections;
}

std::vector<std::vector<cv::Point2f>> gen_selection_vectors(std::vector<std::vector<cv::Point2f>> vvP2D)
{
    std::vector<std::vector<cv::Point2f>> selections;
    selections.reserve(vvP2D.size());

    for (int i = 0; i < vvP2D.size(); i++)
    {
        std::vector<cv::Point2f> v(vvP2D[i].size());
        selections.push_back(v);
    }
    return selections;
}

std::vector<std::vector<cv::Point3f>> gen_selection_vectors(std::vector<std::vector<cv::Point3f>> vvP3Dw)
{
    std::vector<std::vector<cv::Point3f>> selections;
    selections.reserve(vvP3Dw.size());

    for (int i = 0; i < vvP3Dw.size(); i++)
    {
        std::vector<cv::Point3f> v(vvP3Dw[i].size());
        selections.push_back(v);
    }
    return selections;
}

void AutoCalibration::select_random(std::vector<bool>& selection, int n)
{
    if (selection.size() <= n)
    {
        for (int i = 0; i < selection.size(); i++)
        {
            selection[i] = true;
        }
        return;
    }
    for (int i = 0; i < n; i++)
    {
        int idx = (n * rand()) / RAND_MAX;
        if (selection[idx])
            continue;
        selection[idx] = true;
    }
}

void AutoCalibration::select_random(std::vector<std::vector<bool>>& selections, int n)
{
    for (std::vector<bool>& selection : selections)
    {
        if (selection.size() <= n)
        {
            for (int i = 0; i < selection.size(); i++)
            {
                selection[i] = true;
            }
            return;
        }

        for (int i = 0; i < n; i++)
        {
            int idx = rand() % selection.size();
            if (selection[idx])
            {
                i--;
                continue;
            }
            selection[idx] = true;
        }
    }
}

void AutoCalibration::pick_selection(std::vector<cv::Point2f>& skp,
                                     std::vector<cv::Point3f>& swp,
                                     std::vector<cv::Point2f>& nskp,
                                     std::vector<cv::Point3f>& nswp,
                                     std::vector<bool>& selection,
                                     std::vector<cv::Point2f>& keypoints,
                                     std::vector<cv::Point3f>& worldpoints)
{
    for (int i = 0; i < selection.size(); i++)
    {
        if (selection[i])
        {
            skp.push_back(keypoints[i]);
            swp.push_back(worldpoints[i]);
            selection[i] = 0;
        }
        else
        {
            nskp.push_back(keypoints[i]);
            nswp.push_back(worldpoints[i]);
        }
    }
}

void AutoCalibration::pick_selection(std::vector<std::vector<cv::Point2f>>& skp,
                                     std::vector<std::vector<cv::Point3f>>& swp,
                                     std::vector<std::vector<cv::Point2f>>& nskp,
                                     std::vector<std::vector<cv::Point3f>>& nswp,
                                     std::vector<std::vector<bool>>&        selections,
                                     std::vector<std::vector<cv::Point2f>>& keypoints,
                                     std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    for (int i = 0; i < selections.size(); i++)
    {
        for (int j = 0; j < selections[i].size(); j++)
        {
            if (selections[i][j])
            {
                skp[i].push_back(keypoints[i][j]);
                swp[i].push_back(worldpoints[i][j]);
                selections[i][j] = 0;
            }
            else
            {
                nskp[i].push_back(keypoints[i][j]);
                nswp[i].push_back(worldpoints[i][j]);
            }
        }
    }
}

bool AutoCalibration::calibrate_ransac(cv::Mat& intrinsic, cv::Mat& distortion, cv::Size& size,
                                       float& total_error, int nb_iter, int percent_correct,
                                       float threshold, int nselect,
                                       std::vector<cv::Point2f>& keypoints,
                                       std::vector<cv::Point3f>& worldpoints)
{
    int     d           = (percent_correct / 100.0) * worldpoints.size();
    cv::Mat rvec, tvec;
    int     N = keypoints.size();

    //Create temporary vectors
    std::vector<bool>      selection(N, 0);
    std::vector<cv::Point2f> skp;
    std::vector<cv::Point3f> swp;
    std::vector<cv::Point2f> nskp;
    std::vector<cv::Point3f> nswp;
    std::vector<cv::Point2f> projected;
    skp.reserve(N);
    swp.reserve(N);
    nskp.reserve(N);
    nswp.reserve(N);
    projected.reserve(N);

    //Save of initial model parameters
    cv::Mat ini_matrix     = intrinsic.clone();
    cv::Mat ini_distortion = distortion.clone();

    //Model parameters to optimize
    cv::Mat matrix;
    cv::Mat distort;

    for (int k = 0; k < nb_iter; k++)
    {
        //Clear temporary Vectors
        skp.clear();
        swp.clear();

        //Select nselect of all points
        select_random(selection, nselect);
        pick_selection(skp, swp, nskp, nswp, selection, keypoints, worldpoints);

        //Get model with selected points
        matrix                 = ini_matrix.clone();
        distort                = ini_distortion.clone();
        float curr_total_error = calibrate_opencv(matrix, distort, size, rvec, tvec, keypoints, worldpoints);

        //apply model on complement
        cv::projectPoints(nswp, rvec, tvec, matrix, distort, projected);

        //add good points to the current model
        for (int i = 0; i < nswp.size(); i++)
        {
            cv::Point2f p = nskp[i] - projected[i];
            float       t = sqrt(p.x * p.x + p.y * p.y);
            if (t < threshold)
            {
                skp.push_back(nskp[i]);
                swp.push_back(nswp[i]);

                // Add current point error to total current error
                curr_total_error += t;
            }
        }

        curr_total_error = curr_total_error / skp.size();

        // If there is more than d elements that fit the current model and
        //  the total error with this model is the lowest we have so far => keeps this model
        if (skp.size() > d && curr_total_error < total_error)
        {
            float fy  = matrix.at<double>(1, 1);
            float cy  = matrix.at<double>(1, 2);
            float fov = 2.0 * atan2(cy, fy);

            if (fov > 30 && fov < 50)
            {
                total_error = curr_total_error;
                intrinsic   = matrix.clone();
                distortion  = distort.clone();
            }
        }
    }

    return total_error;
}

bool AutoCalibration::calibrate_ransac(cv::Mat& intrinsic, cv::Mat& distortion, cv::Size& size,
                                       float& total_error, int nb_iter, int percent_correct,
                                       float threshold, int nselect,
                                       std::vector<std::vector<cv::Point2f>>& keypoints,
                                       std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    bool has_solution = false;
    int  N            = keypoints.size();

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
    cv::Mat ini_distortion = distortion.clone();

    //Model parameters to optimize
    cv::Mat              matrix;
    std::vector<cv::Mat> rvecs, tvecs;
    tvecs.reserve(N);
    rvecs.reserve(N);

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
        tvecs.clear();
        rvecs.clear();

        //Select nselect of all points
        select_random(selections, nselect);
        pick_selection(skp, swp, nskp, nswp, selections, keypoints, worldpoints);

        // Compute parameters for nselect points per frame
        float err;
        if (!calibrateBruteForce(matrix, skp, swp, rvecs, tvecs, err))
        {
            k = k-1;
            continue;
        }

        cout << calcCameraFOV(matrix) << endl;

        float curr_total_error = 0;

        int nb_correct = 0;
        for (int i = 0; i < N; i++)
        {
            projected.clear();
            //apply model on complement
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

        //cout << "nb correct " << nb_correct << "curr total error " << total_error << endl;
        if (nb_correct >= d && curr_total_error < total_error)
        {
            float fy  = matrix.at<double>(1, 1);
            float cy  = matrix.at<double>(1, 2);
            float fov = 360.0 * atan2(cy, fy) / M_PI;

            if (fov > 30 && fov < 50)
            {
                total_error  = curr_total_error;
                intrinsic    = matrix.clone();
                has_solution = true;
            }
        }
    }
    return has_solution;
};

void mean_position(cv::Point3f& mean, cv::Point3f& max, cv::Point3f& min, std::vector<cv::Point3f>& points3d)
{
    mean = cv::Point3f(0, 0, 0);
    min  = cv::Point3f(99, 99, 99);
    max  = cv::Point3f(-99, -99, -99);

    for (cv::Point3f v : points3d)
    {
        mean += v * (1.0 / points3d.size());
        if (v.x < min.x) { min.x = v.x; }
        if (v.y < min.y) { min.y = v.y; }
        if (v.z < min.z) { min.z = v.z; }

        if (v.x > max.x) { max.x = v.x; }
        if (v.y > max.y) { max.y = v.y; }
        if (v.z > max.z) { max.z = v.z; }
    }
}

void AutoCalibration::savePoints()
{
    static int n;
    std::cout << "save correspondances " << endl;

    for (int i = 0; i < _vvP3Dw.size(); i++)
    {
        std::stringstream ss;
        ss << "worldpoints_" << (n);
        std::string wppath = ss.str();

        ss.str("");
        ss << "keypoints_" << (n++);
        std::string kppath = ss.str();

        ofstream file;
        file.open(wppath);
        for (cv::Point3f p : _vvP3Dw[i])
            file << p.x << " " << p.y << " " << p.z << endl;
        file.close();

        file.open(kppath);
        for (cv::Point2f p : _vvP2D[i])
            file << p.x << " " << p.y << endl;
        file.close();
    }
}

float AutoCalibration::reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
                                       std::vector<cv::Point2f>& vP2D,
                                       std::vector<cv::Point3f>& vP3Dw,
                                       const cv::Mat& rvec,
                                       const cv::Mat& tvec)
{
    std::vector <cv::Point2f> projected;
    cv::projectPoints(vP3Dw, rvec, tvec, intrinsic, distortion, projected);
    return norm(vP2D, projected, NORM_L2) / vP2D.size();
}


float AutoCalibration::reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
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

        error += norm(vvP2D[i], projected, NORM_L2);
        n = n + vvP2D[i].size();
    }
    return error / n;
}

