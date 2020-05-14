#include <WAIAutoCalibration.h>
#include <Utils.h>
using namespace cv;
#define NB_SAMPLES 10

float AutoCalibration::calibrate_opencv(cv::Mat& intrinsic, 
                                        cv::Mat& distortion,
                                        cv::Size& size,
                                        std::vector<cv::Mat>& rvecs,
                                        std::vector<cv::Mat>& tvecs,
                                        std::vector<std::vector<cv::Point2f>>& keypoints,
                                        std::vector<std::vector<cv::Point3f>>& worldpoints)
{

    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
    cv::calibrateCamera(worldpoints, 
                        keypoints, 
                        size, 
                        intrinsic, 
                        distortion, 
                        rvecs, 
                        tvecs,
                        cv::CALIB_USE_INTRINSIC_GUESS |
                        cv::CALIB_ZERO_TANGENT_DIST | 
                        cv::CALIB_FIX_ASPECT_RATIO |
                        cv::CALIB_FIX_PRINCIPAL_POINT |
                        cv::CALIB_FIX_S1_S2_S3_S4 |
                        cv::CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6);

    distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    std::vector <cv::Point2f> projected;
    float error = 0;
    int n = 0;
    for (int i = 0; i < worldpoints.size(); i++)
    {
        cv::projectPoints(worldpoints[i], rvecs[i], tvecs[i], intrinsic, distortion, projected);

        error += norm(keypoints[i], projected, NORM_L2);
        n = n + keypoints[i].size();
    }
    return error / n;
}

void AutoCalibration::select_random(std::vector<bool>& selection, int n)
{
    if (selection.size() <= n)
    {
        for (int i = 0; i < selection.size(); i++)
            selection[i] = true;
        return;
    }
    for (int i = 0; i < n; i++)
    {
        int idx = rand() % n;
        if (selection[idx])
            continue;

        selection[idx] = true;
    }
}

void AutoCalibration::pick_frames(std::vector<bool>&                     selections,
                                  std::vector<std::vector<cv::Point2f>>& skp,
                                  std::vector<std::vector<cv::Point3f>>& swp,
                                  std::vector<std::vector<cv::Point2f>>& nskp,
                                  std::vector<std::vector<cv::Point3f>>& nswp,
                                  std::vector<std::vector<cv::Point2f>>& keypoints,
                                  std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    for (int i = 0; i < selections.size(); i++)
    {
        if (selections[i])
        {
            skp.push_back(keypoints[i]);
            swp.push_back(worldpoints[i]);
            selections[i] = 0;
        }
        else
        {
            nskp.push_back(keypoints[i]);
            nswp.push_back(worldpoints[i]);
        }
    }
}

void AutoCalibration::pick_points(std::vector<bool>&        selections,
                                  std::vector<cv::Point2f>& skp,
                                  std::vector<cv::Point3f>& swp,
                                  std::vector<cv::Point2f>& nskp,
                                  std::vector<cv::Point3f>& nswp,
                                  std::vector<cv::Point2f>& keypoints,
                                  std::vector<cv::Point3f>& worldpoints)
{
    for (int i = 0; i < selections.size(); i++)
    {
        if (selections[i])
        {
            skp.push_back(keypoints[i]);
            swp.push_back(worldpoints[i]);
            selections[i] = 0;
        }
        else
        {
            nskp.push_back(keypoints[i]);
            nswp.push_back(worldpoints[i]);
        }
    }
}

void AutoCalibration::computeMatrix(cv::Size size, cv::Mat& mat, cv::Mat &distortion, float fov)
{
    float cx   = (float)size.width * 0.5f;
    float cy   = (float)size.height * 0.5f;
    float fx   = cx / tanf(fov * 0.5f * M_PI / 180.0);
    float fy   = fx;
    mat        = (Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distortion = (Mat_<float>(1, 5) << 0, 0, 0, 0, 0);
}

bool AutoCalibration::calibrateBruteForce(cv::Mat&                               intrinsic,
                                          std::vector<std::vector<cv::Point2f>>& vvP2D,
                                          std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                          std::vector<cv::Mat>&                  rvecs,
                                          std::vector<cv::Mat>&                  tvecs,
                                          cv::Size                               size,
                                          float&                                 error)
{
    bool ret = false;
    error = 999999999.0;

    for (int i = 0; i < 6; i++)
    {
        float fov = 57 + 3 * i;
        std::vector<cv::Mat> rotvecs, trvecs;
        cv::Mat matrix;
        cv::Mat distortion;
        computeMatrix(size, matrix, distortion, fov);

        float err = calibrate_opencv(matrix, distortion, size, rotvecs, trvecs, vvP2D, vvP3Dw);

        float fx = matrix.at<double>(0, 0);
        float cx = matrix.at<double>(0, 2);
        float hfov = 2.0 * atan2(cx, fx) * 180.0 / M_PI;

        if (err < error && hfov > 50 && hfov < 90)
        {
            error     = err;
            intrinsic = matrix.clone();
            rvecs     = rotvecs;
            tvecs     = trvecs;
            ret       = true;
        }
    }
    return ret;
}

bool AutoCalibration::calibrateBruteForce(cv::Mat &intrinsic,
                                          std::vector<cv::Point2f>& vP2D,
                                          std::vector<cv::Point3f>& vP3Dw,
                                          cv::Mat& rvec,
                                          cv::Mat& tvec,
                                          cv::Size size,
                                          float &error)
{
    bool ret = false;
    error = 999999999.0;

    for (int i = 0; i < 6; i++)
    {
        float fov = 57 + 3 * i;
        std::vector<cv::Mat> rotvecs, trvecs;
        rotvecs.resize(1);
        trvecs.resize(1);

        std::vector<std::vector<cv::Point2f>> vvP2D;
        vvP2D.push_back(vP2D);
        std::vector<std::vector<cv::Point3f>> vvP3Dw;
        vvP3Dw.push_back(vP3Dw);

        cv::Mat matrix;
        cv::Mat distortion;
        computeMatrix(size, matrix, distortion, fov);
        float err = calibrate_opencv(matrix, distortion, size, rotvecs, trvecs, vvP2D, vvP3Dw);

        float fx = matrix.at<double>(0, 0);
        float cx = matrix.at<double>(0, 2);
        float hfov = 2.0 * atan2(cx, fx) * 180.0 / M_PI;

        if (err < error && hfov > 50 && hfov < 90)
        {
            error     = err;
            intrinsic = matrix.clone();
            rvec      = rotvecs[0];
            tvec      = trvecs[0];
            ret       = true;
        }
    }
    return ret;
}

float AutoCalibration::ransac_frame_points(cv::Size&                 size,
                                           int                       nbIter,
                                           float                     threshold,
                                           int                       iniModelSize,
                                           std::vector<cv::Point2f>& keypoints,
                                           std::vector<cv::Point3f>& worldpoints,
                                           std::vector<cv::Point2f>& outKeypoints,
                                           std::vector<cv::Point3f>& outWorldpoints)
{
    int     N = keypoints.size();
    cv::Mat rvec, tvec;

    //Create temporary vectors
    std::vector<cv::Point2f> skp;
    std::vector<cv::Point3f> swp;
    std::vector<cv::Point2f> nskp;
    std::vector<cv::Point3f> nswp;
    std::vector<cv::Point2f> projected;
    std::vector<bool>        selection;

    selection.resize(N);
    skp.reserve(N);
    swp.reserve(N);
    nskp.reserve(N);
    nswp.reserve(N);
    projected.reserve(N);

    //Model parameters to optimize
    cv::Mat matrix;
    float total_error = 999999999.0;

    float error;

    for (int k = 0; k < nbIter; k++)
    {
        //Clear temporary Vectors
        skp.clear();
        swp.clear();
        nskp.clear();
        nswp.clear();

        //Select nselect of all points
        select_random(selection, iniModelSize);
        pick_points(selection, skp, swp, nskp, nswp, keypoints, worldpoints);

        // Compute parameters for selected model
        if (skp.size() < 6)
            continue;

        if (!calibrateBruteForce(matrix, skp, swp, rvec, tvec, size, error))
            continue;

        //add complement points that match the current model

        cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
        cv::projectPoints(nswp, rvec, tvec, matrix, distortion, projected);

        for (int i = 0; i < nswp.size(); i++)
        {
            cv::Point2f p = nskp[i] - projected[i];
            float       t = sqrt(p.x * p.x + p.y * p.y);

            if (t < threshold)
            {
                skp.push_back(nskp[i]);
                swp.push_back(nswp[i]);
            }
        }

        if (!calibrateBruteForce(matrix, skp, swp, rvec, tvec, size, error))
            continue;

        float fx  = matrix.at<double>(0, 0);
        float cx  = matrix.at<double>(0, 2);
        float hfov = 2.0 * atan2(cx, fx) * 180.0 / M_PI;

        // If there is more than d elements that fit the current model and
        //  the total error with this model is the lowest we have so far => keeps this model
        if (error < total_error && hfov > 50 && hfov < 90)
        {
            total_error    = error;
            outKeypoints   = skp;
            outWorldpoints = swp;
        }
    }

    return total_error;
}

float AutoCalibration::calibrate_frames_ransac(cv::Size&                              size,
                                               cv::Mat&                               intrinsic,
                                               int                                    nbIter,
                                               float                                  threshold,
                                               int                                    iniModelSize,
                                               std::vector<std::vector<cv::Point2f>>& keypoints,
                                               std::vector<std::vector<cv::Point3f>>& worldpoints,
                                               int nbVectors)
{
    int  N            = nbVectors;

    std::vector<std::vector<cv::Point2f>> skp;  //maybe model
    std::vector<std::vector<cv::Point3f>> swp;
    std::vector<std::vector<cv::Point2f>> nskp; //maybe inlier
    std::vector<std::vector<cv::Point3f>> nswp;
    std::vector<bool> selections;
    selections.resize(N);

    //Model parameters to optimize
    cv::Mat              matrix;
    std::vector<cv::Mat> rvecs, tvecs;
    float                total_error = 999999999.0;
    float                error       = 0;


    for (int k = 0; k < nbIter; k++)
    {
        swp.clear();
        skp.clear();
        nswp.clear();
        nskp.clear();

        select_random(selections, iniModelSize);
        pick_frames(selections, skp, swp, nskp, nswp, keypoints, worldpoints);

        // Compute parameters for selected model
        float error;
        if (!calibrateBruteForce(matrix, skp, swp, rvecs, tvecs, size, error))
            continue;

        // Add complement points that match the current model
        for (int i = 0; i < nswp.size(); i++)
        {
            std::vector<cv::Point2f> projected;
            cv::Mat rvec, tvec;
            cv::Mat distortion = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
            cv::solvePnP(nswp[i], nskp[i], matrix, distortion, rvec, tvec);
            cv::projectPoints(nswp[i], rvec, tvec, matrix, distortion, projected);

            for (int j = 0; j < nskp[i].size(); j++)
            {
                cv::Point2f p = nskp[i][j] - projected[j];
                float       t = sqrt(p.x * p.x + p.y * p.y);
                error += t / nskp[i].size();
            }
            if (error < threshold)
            {
                swp.push_back(nswp[i]);
                skp.push_back(nskp[i]);
            }
        }

        //Compute error on complete model
        calibrateBruteForce(matrix, skp, swp, rvecs, tvecs, size, error);

        float fx  = matrix.at<double>(0, 0);
        float cx  = matrix.at<double>(0, 2);
        float hfov = 2.0 * atan2(cx, fx) * 180.0 / M_PI;

        // If the total error with this model is the lowest we have so far => keeps this model
        if (error < total_error && hfov > 50 && hfov < 90)
        {
            if (hfov > 50 && hfov < 90)
            {
                total_error  = error;
                intrinsic    = matrix.clone();
            }
        }
    }
    return total_error;
};

float AutoCalibration::calcCameraVerticalFOV(cv::Mat& cameraMat)
{
    float fy = (float)cameraMat.at<double>(1, 1);
    float cy = (float)cameraMat.at<double>(1, 2);
    return 2.0 * atan2(cy, fy) * 180.0 / M_PI;
}

float AutoCalibration::calcCameraHorizontalFOV(cv::Mat& cameraMat)
{
    float fx = (float)cameraMat.at<double>(0, 0);
    float cx = (float)cameraMat.at<double>(0, 2);
    return 2.0 * atan2(cx, fx) * 180.0 / M_PI;
}

void AutoCalibration::calibrate(cv::Size size,
                                std::vector<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>> matchings)
{
    std::vector<std::vector<cv::Point2f>> preselectedKeyPoints;
    std::vector<std::vector<cv::Point3f>> preselectedWorldPoints;

    preselectedKeyPoints.resize(matchings.size());
    preselectedWorldPoints.resize(matchings.size());
    int nbFill = 0;

    for (int i = 0; i < matchings.size(); i++)
    {
        std::vector<cv::Point2f> p2f = matchings[i].first;
        std::vector<cv::Point3f> p3f = matchings[i].second;

        float error = ransac_frame_points(size,
                                          10,
                                          3.0, //threshold
                                          p2f.size() * 0.4,
                                          p2f,
                                          p3f,
                                          preselectedKeyPoints[nbFill],
                                          preselectedWorldPoints[nbFill]);

        if (preselectedKeyPoints[nbFill].size() > 6)
            nbFill++;
    }

    std::cout << "nbFill " << nbFill << std::endl;
    if (nbFill < 3)
        return;

    cv::Mat intrinsic;

    float error = calibrate_frames_ransac(size,
                                          intrinsic,
                                          10,
                                          3.0, //threshold
                                          2,
                                          preselectedKeyPoints,
                                          preselectedWorldPoints,
                                          nbFill);

    matchings.clear();

    float fy = (float)intrinsic.at<double>(1, 1);
    float cy = (float)intrinsic.at<double>(1, 2);
    float fx = (float)intrinsic.at<double>(0, 0);
    float cx = (float)intrinsic.at<double>(0, 2);

    Utils::log("Info", "AAAAAAAA vfov %f\n", calcCameraVerticalFOV(intrinsic));
    Utils::log("Info", "AAAAAAAA hfov %f\n", calcCameraHorizontalFOV(intrinsic));
    Utils::log("Info", "AAAAAAAA fx fy cx cy %f %f %f %f\n", fx, fy, cx, cy);
}
