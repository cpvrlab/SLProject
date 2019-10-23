
#include "WAIOrbSlamInitialization.h"

typedef std::pair<i32, i32> Match;

void decomposeE(const cv::Mat& E,
                cv::Mat&       R1,
                cv::Mat&       R2,
                cv::Mat&       t)
{
    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

void triangulate(const cv::KeyPoint& kp1,
                 const cv::KeyPoint& kp2,
                 const cv::Mat&      P1,
                 const cv::Mat&      P2,
                 cv::Mat&            x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<r32>(3);
}

i32 checkRT(const cv::Mat&                   R,
            const cv::Mat&                   t,
            const std::vector<cv::KeyPoint>& keyPointsKeyFrame1,
            const std::vector<cv::KeyPoint>& keyPointsKeyFrame2,
            const cv::Mat&                   descriptors1,
            const cv::Mat&                   descriptors2,
            const std::vector<Match>&        matches,
            std::vector<bool32>&             vbMatchesInliers,
            const cv::Mat&                   cameraMat,
            std::vector<cv::Point3f>&        pointCandidates,
            r32                              th2,
            std::vector<bool32>&             vbGood,
            r32&                             parallax)
{
    // Calibration parameters
    const r32 fx = cameraMat.at<r32>(0, 0);
    const r32 fy = cameraMat.at<r32>(1, 1);
    const r32 cx = cameraMat.at<r32>(0, 2);
    const r32 cy = cameraMat.at<r32>(1, 2);

    vbGood = std::vector<bool32>(keyPointsKeyFrame1.size(), false);
    pointCandidates.resize(keyPointsKeyFrame1.size());

    std::vector<r32> vCosParallax;
    vCosParallax.reserve(keyPointsKeyFrame1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    cameraMat.copyTo(P1.rowRange(0, 3).colRange(0, 3));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = cameraMat * P2;

    cv::Mat O2 = -R.t() * t;

    i32 nGood = 0;

    for (size_t i = 0, iend = matches.size(); i < iend; i++)
    {
        if (!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint& kp1 = keyPointsKeyFrame1[matches[i].first];
        const cv::KeyPoint& kp2 = keyPointsKeyFrame2[matches[i].second];
        cv::Mat             p3dC1;

        triangulate(kp1, kp2, P1, P2, p3dC1);

        if (!std::isfinite(p3dC1.at<r32>(0)) ||
            !std::isfinite(p3dC1.at<r32>(1)) ||
            !std::isfinite(p3dC1.at<r32>(2)))
        {
            vbGood[matches[i].first] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        r32     dist1   = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        r32     dist2   = cv::norm(normal2);

        r32 cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<r32>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R * p3dC1 + t;

        if (p3dC2.at<r32>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        r32 im1x, im1y;
        r32 invZ1 = 1.0 / p3dC1.at<r32>(2);
        im1x      = fx * p3dC1.at<r32>(0) * invZ1 + cx;
        im1y      = fy * p3dC1.at<r32>(1) * invZ1 + cy;

        r32 squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

        if (squareError1 > th2)
            continue;

        // Check reprojection error in second image
        r32 im2x, im2y;
        r32 invZ2 = 1.0 / p3dC2.at<r32>(2);
        im2x      = fx * p3dC2.at<r32>(0) * invZ2 + cx;
        im2y      = fy * p3dC2.at<r32>(1) * invZ2 + cy;

        r32 squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

        if (squareError2 > th2)
            continue;

        vCosParallax.push_back(cosParallax);
        pointCandidates[matches[i].first] = cv::Point3f(p3dC1.at<r32>(0), p3dC1.at<r32>(1), p3dC1.at<r32>(2));
        nGood++;

        if (cosParallax < 0.99998)
            vbGood[matches[i].first] = true;
    }

    if (nGood > 0)
    {
        sort(vCosParallax.begin(), vCosParallax.end());

        size_t idx = std::min(50, i32(vCosParallax.size() - 1));
        parallax   = acos(vCosParallax[idx]) * 180 / CV_PI;
    }
    else
    {
        parallax = 0;
    }

    return nGood;
}

bool reconstructFundamental(const std::vector<Match>&        matches,
                            const std::vector<cv::KeyPoint>& keyPoints1,
                            const std::vector<cv::KeyPoint>& keyPoints2,
                            const cv::Mat&                   descriptors1,
                            const cv::Mat&                   descriptors2,
                            const r32                        sigma,
                            std::vector<bool32>&             vbMatchesInliers,
                            cv::Mat&                         F21,
                            const cv::Mat&                   cameraMat,
                            cv::Mat&                         R21,
                            cv::Mat&                         t21,
                            std::vector<cv::Point3f>&        initialPoints,
                            std::vector<bool32>&             vbTriangulated,
                            float                            minParallax,
                            int                              minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
        if (vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = cameraMat.t() * F21 * cameraMat;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    decomposeE(E21, R1, R2, t);

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    std::vector<bool32>      vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float                    parallax1, parallax2, parallax3, parallax4;

    r32 sigmaSq = sigma * sigma;
    int nGood1  = checkRT(R1,
                         t1,
                         keyPoints1,
                         keyPoints2,
                         descriptors1,
                         descriptors2,
                         matches,
                         vbMatchesInliers,
                         cameraMat,
                         vP3D1,
                         4.0 * sigmaSq,
                         vbTriangulated1,
                         parallax1);
    int nGood2  = checkRT(R2,
                         t1,
                         keyPoints1,
                         keyPoints2,
                         descriptors1,
                         descriptors2,
                         matches,
                         vbMatchesInliers,
                         cameraMat,
                         vP3D2,
                         4.0 * sigmaSq,
                         vbTriangulated2,
                         parallax2);
    int nGood3  = checkRT(R1,
                         t2,
                         keyPoints1,
                         keyPoints2,
                         descriptors1,
                         descriptors2,
                         matches,
                         vbMatchesInliers,
                         cameraMat,
                         vP3D3,
                         4.0 * sigmaSq,
                         vbTriangulated3,
                         parallax3);
    int nGood4  = checkRT(R2,
                         t2,
                         keyPoints1,
                         keyPoints2,
                         descriptors1,
                         descriptors2,
                         matches,
                         vbMatchesInliers,
                         cameraMat,
                         vP3D4,
                         4.0 * sigmaSq,
                         vbTriangulated4,
                         parallax4);

    int maxGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = std::max(static_cast<int>(0.9 * N), minTriangulated);

    int nsimilar = 0;
    if (nGood1 > 0.7 * maxGood)
        nsimilar++;
    if (nGood2 > 0.7 * maxGood)
        nsimilar++;
    if (nGood3 > 0.7 * maxGood)
        nsimilar++;
    if (nGood4 > 0.7 * maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if (maxGood < nMinGood || nsimilar > 1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if (maxGood == nGood1)
    {
        if (parallax1 > minParallax)
        {
            initialPoints = vP3D1;

            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood2)
    {
        if (parallax2 > minParallax)
        {
            initialPoints = vP3D2;

            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood3)
    {
        if (parallax3 > minParallax)
        {
            initialPoints = vP3D3;

            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood4)
    {
        if (parallax4 > minParallax)
        {
            initialPoints = vP3D4;

            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool32 reconstructHomography(const std::vector<Match>&        matches,
                             const std::vector<cv::KeyPoint>& keyPoints1,
                             const std::vector<cv::KeyPoint>& keyPoints2,
                             const cv::Mat&                   keyPointDescriptors1,
                             const cv::Mat&                   keyPointDescriptors2,
                             const r32                        sigma,
                             std::vector<bool32>&             vbMatchesInliers,
                             cv::Mat&                         H21,
                             const cv::Mat&                   cameraMat,
                             cv::Mat&                         R21,
                             cv::Mat&                         t21,
                             std::vector<cv::Point3f>&        initialPoints,
                             std::vector<bool32>&             vbTriangulated,
                             r32                              minParallax,
                             i32                              minTriangulated)
{
    i32 N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
    {
        if (vbMatchesInliers[i])
        {
            N++;
        }
    }

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = cameraMat.inv();
    cv::Mat A    = invK * H21 * cameraMat;

    cv::Mat U, w, Vt, V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    r32 s = cv::determinant(U) * cv::determinant(Vt);

    r32 d1 = w.at<r32>(0);
    r32 d2 = w.at<r32>(1);
    r32 d3 = w.at<r32>(2);

    if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
    {
        return false;
    }

    std::vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    r32 aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
    r32 aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
    r32 x1[] = {aux1, aux1, -aux1, -aux1};
    r32 x3[] = {aux3, -aux3, aux3, -aux3};

    //case d'=d2
    r32 aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

    r32 ctheta   = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    r32 stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for (i32 i = 0; i < 4; i++)
    {
        cv::Mat Rp       = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<r32>(0, 0) = ctheta;
        Rp.at<r32>(0, 2) = -stheta[i];
        Rp.at<r32>(2, 0) = stheta[i];
        Rp.at<r32>(2, 2) = ctheta;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<r32>(0) = x1[i];
        tp.at<r32>(1) = 0;
        tp.at<r32>(2) = -x3[i];
        tp *= d1 - d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<r32>(0) = x1[i];
        np.at<r32>(1) = 0;
        np.at<r32>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<r32>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    r32 aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

    r32 cphi   = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    r32 sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for (i32 i = 0; i < 4; i++)
    {
        cv::Mat Rp       = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<r32>(0, 0) = cphi;
        Rp.at<r32>(0, 2) = sphi[i];
        Rp.at<r32>(1, 1) = -1;
        Rp.at<r32>(2, 0) = sphi[i];
        Rp.at<r32>(2, 2) = -cphi;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<r32>(0) = x1[i];
        tp.at<r32>(1) = 0;
        tp.at<r32>(2) = x3[i];
        tp *= d1 + d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<r32>(0) = x1[i];
        np.at<r32>(1) = 0;
        np.at<r32>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<r32>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    i32                      bestGood        = 0;
    i32                      secondBestGood  = 0;
    i32                      bestSolutionIdx = -1;
    r32                      bestParallax    = -1;
    std::vector<cv::Point3f> bestInitialPoints;
    std::vector<bool32>      bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for (size_t i = 0; i < 8; i++)
    {
        r32                      parallaxi;
        std::vector<cv::Point3f> pointCandidates;
        std::vector<bool32>      vbTriangulatedi;

        i32 nGood =
          checkRT(vR[i],
                  vt[i],
                  keyPoints1,
                  keyPoints2,
                  keyPointDescriptors1,
                  keyPointDescriptors2,
                  matches,
                  vbMatchesInliers,
                  cameraMat,
                  pointCandidates,
                  4.0 * (sigma * sigma),
                  vbTriangulatedi,
                  parallaxi);

        if (nGood > bestGood)
        {
            secondBestGood    = bestGood;
            bestGood          = nGood;
            bestSolutionIdx   = i;
            bestParallax      = parallaxi;
            bestInitialPoints = pointCandidates;
            bestTriangulated  = vbTriangulatedi;
        }
        else if (nGood > secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9 * N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        initialPoints  = bestInitialPoints;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

r32 checkFundamental(const std::vector<Match>&        matches,
                     const std::vector<cv::KeyPoint>& keyPoints1,
                     const std::vector<cv::KeyPoint>& keyPoints2,
                     const cv::Mat&                   F21,
                     const r32                        sigma,
                     std::vector<bool32>&             vbMatchesInliers)
{
    const i32 N = matches.size();

    const r32 f11 = F21.at<r32>(0, 0);
    const r32 f12 = F21.at<r32>(0, 1);
    const r32 f13 = F21.at<r32>(0, 2);
    const r32 f21 = F21.at<r32>(1, 0);
    const r32 f22 = F21.at<r32>(1, 1);
    const r32 f23 = F21.at<r32>(1, 2);
    const r32 f31 = F21.at<r32>(2, 0);
    const r32 f32 = F21.at<r32>(2, 1);
    const r32 f33 = F21.at<r32>(2, 2);

    vbMatchesInliers.resize(N);

    r32 score = 0;

    const r32 th      = 3.841;
    const r32 thScore = 5.991;

    const r32 invSigmaSquare = 1.0 / (sigma * sigma);

    for (i32 i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint& kp1 = keyPoints1[matches[i].first];
        const cv::KeyPoint& kp2 = keyPoints2[matches[i].second];

        const r32 u1 = kp1.pt.x;
        const r32 v1 = kp1.pt.y;
        const r32 u2 = kp2.pt.x;
        const r32 v2 = kp2.pt.y;

        const r32 a2 = f11 * u1 + f12 * v1 + f13;
        const r32 b2 = f21 * u1 + f22 * v1 + f23;
        const r32 c2 = f31 * u1 + f32 * v1 + f33;

        const r32 num2 = a2 * u2 + b2 * v2 + c2;

        const r32 squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const r32 chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        const r32 a1 = f11 * u2 + f21 * v2 + f31;
        const r32 b1 = f12 * u2 + f22 * v2 + f32;
        const r32 c1 = f13 * u2 + f23 * v2 + f33;

        const r32 num1 = a1 * u1 + b1 * v1 + c1;

        const r32 squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const r32 chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

r32 checkHomography(const std::vector<Match>&        matches,
                    const std::vector<cv::KeyPoint>& keyPoints1,
                    const std::vector<cv::KeyPoint>& keyPoints2,
                    const cv::Mat&                   H21,
                    const cv::Mat&                   H12,
                    std::vector<bool32>&             vbMatchesInliers,
                    r32                              sigma)
{
    const i32 N = matches.size();

    const r32 h11 = H21.at<r32>(0, 0);
    const r32 h12 = H21.at<r32>(0, 1);
    const r32 h13 = H21.at<r32>(0, 2);
    const r32 h21 = H21.at<r32>(1, 0);
    const r32 h22 = H21.at<r32>(1, 1);
    const r32 h23 = H21.at<r32>(1, 2);
    const r32 h31 = H21.at<r32>(2, 0);
    const r32 h32 = H21.at<r32>(2, 1);
    const r32 h33 = H21.at<r32>(2, 2);

    const r32 h11inv = H12.at<r32>(0, 0);
    const r32 h12inv = H12.at<r32>(0, 1);
    const r32 h13inv = H12.at<r32>(0, 2);
    const r32 h21inv = H12.at<r32>(1, 0);
    const r32 h22inv = H12.at<r32>(1, 1);
    const r32 h23inv = H12.at<r32>(1, 2);
    const r32 h31inv = H12.at<r32>(2, 0);
    const r32 h32inv = H12.at<r32>(2, 1);
    const r32 h33inv = H12.at<r32>(2, 2);

    vbMatchesInliers.resize(N);

    r32 score = 0;

    const r32 th = 5.991;

    const r32 invSigmaSquare = 1.0 / (sigma * sigma);

    for (i32 i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint& kp1 = keyPoints1[matches[i].first];
        const cv::KeyPoint& kp2 = keyPoints2[matches[i].second];

        const r32 u1 = kp1.pt.x;
        const r32 v1 = kp1.pt.y;
        const r32 u2 = kp2.pt.x;
        const r32 v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const r32 w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const r32 u2in1    = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const r32 v2in1    = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        const r32 squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        const r32 chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const r32 w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const r32 u1in2    = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const r32 v1in2    = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const r32 squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const r32 chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

cv::Mat computeF21(const std::vector<cv::Point2f>& vP1,
                   const std::vector<cv::Point2f>& vP2)
{
    const i32 N = vP1.size();

    cv::Mat A(N, 9, CV_32F);

    for (i32 i = 0; i < N; i++)
    {
        const r32 u1 = vP1[i].x;
        const r32 v1 = vP1[i].y;
        const r32 u2 = vP2[i].x;
        const r32 v2 = vP2[i].y;

        A.at<r32>(i, 0) = u2 * u1;
        A.at<r32>(i, 1) = u2 * v1;
        A.at<r32>(i, 2) = u2;
        A.at<r32>(i, 3) = v2 * u1;
        A.at<r32>(i, 4) = v2 * v1;
        A.at<r32>(i, 5) = v2;
        A.at<r32>(i, 6) = u1;
        A.at<r32>(i, 7) = v1;
        A.at<r32>(i, 8) = 1;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<r32>(2) = 0;

    cv::Mat result = u * cv::Mat::diag(w) * vt;
    return result;
}

cv::Mat computeH21(const std::vector<cv::Point2f>& vP1,
                   const std::vector<cv::Point2f>& vP2)
{
    const i32 N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);

    for (i32 i = 0; i < N; i++)
    {
        const r32 u1 = vP1[i].x;
        const r32 v1 = vP1[i].y;
        const r32 u2 = vP2[i].x;
        const r32 v2 = vP2[i].y;

        A.at<r32>(2 * i, 0) = 0.0;
        A.at<r32>(2 * i, 1) = 0.0;
        A.at<r32>(2 * i, 2) = 0.0;
        A.at<r32>(2 * i, 3) = -u1;
        A.at<r32>(2 * i, 4) = -v1;
        A.at<r32>(2 * i, 5) = -1;
        A.at<r32>(2 * i, 6) = v2 * u1;
        A.at<r32>(2 * i, 7) = v2 * v1;
        A.at<r32>(2 * i, 8) = v2;

        A.at<r32>(2 * i + 1, 0) = u1;
        A.at<r32>(2 * i + 1, 1) = v1;
        A.at<r32>(2 * i + 1, 2) = 1;
        A.at<r32>(2 * i + 1, 3) = 0.0;
        A.at<r32>(2 * i + 1, 4) = 0.0;
        A.at<r32>(2 * i + 1, 5) = 0.0;
        A.at<r32>(2 * i + 1, 6) = -u2 * u1;
        A.at<r32>(2 * i + 1, 7) = -u2 * v1;
        A.at<r32>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat result = vt.row(8).reshape(0, 3);
    return result;
}

void normalizeKeyPoints(const std::vector<cv::KeyPoint>& vKeys,
                        std::vector<cv::Point2f>&        vNormalizedPoints,
                        cv::Mat&                         T)
{
    r32       meanX = 0;
    r32       meanY = 0;
    const i32 N     = vKeys.size();

    vNormalizedPoints.resize(N);

    for (i32 i = 0; i < N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    r32 meanDevX = 0;
    r32 meanDevY = 0;

    for (i32 i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    r32 sX = 1.0 / meanDevX;
    r32 sY = 1.0 / meanDevY;

    for (i32 i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T               = cv::Mat::eye(3, 3, CV_32F);
    T.at<r32>(0, 0) = sX;
    T.at<r32>(1, 1) = sY;
    T.at<r32>(0, 2) = -meanX * sX;
    T.at<r32>(1, 2) = -meanY * sY;
}

void findFundamental(const std::vector<Match>&               matches,
                     const std::vector<cv::KeyPoint>&        keyPoints1,
                     const std::vector<cv::KeyPoint>&        keyPoints2,
                     const i32                               maxRansacIterations,
                     const std::vector<std::vector<size_t>>& ransacSets,
                     const r32                               sigma,
                     r32&                                    score,
                     std::vector<bool32>&                    inlierMatchesFlags,
                     cv::Mat&                                F21)
{
    // Number of putative matches
    const i32 N = inlierMatchesFlags.size();

    // Normalize coordinates
    std::vector<cv::Point2f> vPn1, vPn2;
    cv::Mat                  T1, T2;
    normalizeKeyPoints(keyPoints1, vPn1, T1);
    normalizeKeyPoints(keyPoints2, vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score              = 0.0;
    inlierMatchesFlags = std::vector<bool32>(N, false);

    // Iteration variables
    std::vector<cv::Point2f> vPn1i(8);
    std::vector<cv::Point2f> vPn2i(8);
    cv::Mat                  F21i;
    std::vector<bool32>      vbCurrentInliers(N, false);
    r32                      currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        // Select a minimum set
        for (i32 j = 0; j < 8; j++)
        {
            i32 idx = ransacSets[it][j];

            vPn1i[j] = vPn1[matches[idx].first];
            vPn2i[j] = vPn2[matches[idx].second];
        }

        cv::Mat Fn = computeF21(vPn1i, vPn2i);

        F21i = T2t * Fn * T1;

        currentScore = checkFundamental(matches,
                                        keyPoints1,
                                        keyPoints2,
                                        F21i,
                                        sigma,
                                        vbCurrentInliers);

        if (currentScore > score)
        {
            F21                = F21i.clone();
            inlierMatchesFlags = vbCurrentInliers;
            score              = currentScore;
        }
    }
}

void findHomography(const std::vector<Match>&               matches,
                    const std::vector<cv::KeyPoint>&        keyPoints1,
                    const std::vector<cv::KeyPoint>&        keyPoints2,
                    const i32                               maxRansacIterations,
                    const std::vector<std::vector<size_t>>& ransacSets,
                    const r32                               sigma,
                    r32&                                    score,
                    std::vector<bool32>&                    inlierMatchesFlags,
                    cv::Mat&                                H21)
{
    // Number of putative matches
    const i32 N = matches.size();

    // Normalize coordinates
    std::vector<cv::Point2f> vPn1, vPn2;
    cv::Mat                  T1, T2;
    normalizeKeyPoints(keyPoints1, vPn1, T1);
    normalizeKeyPoints(keyPoints2, vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score              = 0.0;
    inlierMatchesFlags = std::vector<bool32>(N, false);

    // Iteration variables
    std::vector<cv::Point2f> vPn1i(8);
    std::vector<cv::Point2f> vPn2i(8);
    cv::Mat                  H21i, H12i;
    std::vector<bool32>      vbCurrentInliers(N, false);
    r32                      currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            i32 idx = ransacSets[it][j];

            vPn1i[j] = vPn1[matches[idx].first];
            vPn2i[j] = vPn2[matches[idx].second];
        }

        cv::Mat Hn = computeH21(vPn1i, vPn2i);
        H21i       = T2inv * Hn * T1;
        H12i       = H21i.inv();

        currentScore = checkHomography(matches,
                                       keyPoints1,
                                       keyPoints2,
                                       H21i,
                                       H12i,
                                       vbCurrentInliers,
                                       sigma);

        if (currentScore > score)
        {
            H21                = H21i.clone();
            inlierMatchesFlags = vbCurrentInliers;
            score              = currentScore;
        }
    }
}
