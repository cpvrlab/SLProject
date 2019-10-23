#include "WAIOrbMatching.h"

void computeThreeMaxima(std::vector<i32>* rotationHistory,
                        const i32         historyLength,
                        i32&              ind1,
                        i32&              ind2,
                        i32&              ind3)
{
    i32 max1 = 0;
    i32 max2 = 0;
    i32 max3 = 0;

    for (i32 i = 0; i < historyLength; i++)
    {
        const i32 s = rotationHistory[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (r32)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (r32)max1)
    {
        ind3 = -1;
    }
}

i32 descriptorDistance(const cv::Mat& a,
                       const cv::Mat& b)
{
    const i32* pa = a.ptr<int32_t>();
    const i32* pb = b.ptr<int32_t>();

    i32 dist = 0;

    for (i32 i = 0; i < 8; i++, pa++, pb++)
    {
        u32 v = *pa ^ *pb;
        v     = v - ((v >> 1) & 0x55555555);
        v     = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

bool32 calculateKeyPointGridCell(const cv::KeyPoint&   keyPoint,
                                 const GridConstraints gridConstraints,
                                 i32*                  posX,
                                 i32*                  posY)
{
    bool32 result = false;

    i32 x = (i32)round((keyPoint.pt.x - gridConstraints.minX) * gridConstraints.invGridElementWidth);
    i32 y = (i32)round((keyPoint.pt.y - gridConstraints.minY) * gridConstraints.invGridElementHeight);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (x < 0 || x >= FRAME_GRID_COLS || y < 0 || y >= FRAME_GRID_ROWS)
    {
        result = false;
    }
    else
    {
        *posX  = x;
        *posY  = y;
        result = true;
    }

    return result;
}

void undistortKeyPoints(const cv::Mat                   cameraMat,
                        const cv::Mat                   distortionCoefficients,
                        const std::vector<cv::KeyPoint> keyPoints,
                        const i32                       numberOfKeyPoints,
                        std::vector<cv::KeyPoint>&      undistortedKeyPoints)
{
    if (distortionCoefficients.at<r32>(0) == 0.0f)
    {
        undistortedKeyPoints = keyPoints;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(numberOfKeyPoints, 2, CV_32F);
    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        mat.at<r32>(i, 0) = keyPoints[i].pt.x;
        mat.at<r32>(i, 1) = keyPoints[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cameraMat, distortionCoefficients, cv::Mat(), cameraMat);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    undistortedKeyPoints.resize(numberOfKeyPoints);
    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        cv::KeyPoint kp         = keyPoints[i];
        kp.pt.x                 = mat.at<r32>(i, 0);
        kp.pt.y                 = mat.at<r32>(i, 1);
        undistortedKeyPoints[i] = kp;
    }
}

static std::vector<size_t> getFeatureIndicesForArea(const i32                       numberOfKeyPoints,
                                                    const r32                       searchWindowSize,
                                                    const r32                       x,
                                                    const r32                       y,
                                                    const GridConstraints           gridConstraints,
                                                    const i32                       minLevel,
                                                    const i32                       maxLevel,
                                                    const std::vector<size_t>       keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                                    const std::vector<cv::KeyPoint> undistortedKeyPoints)
{
    std::vector<size_t> result;

    result.reserve(numberOfKeyPoints);

    const i32 nMinCellX = std::max(0, (i32)floor((x - gridConstraints.minX - searchWindowSize) * gridConstraints.invGridElementWidth));
    if (nMinCellX >= FRAME_GRID_COLS)
        return result;

    const i32 nMaxCellX = std::min((i32)FRAME_GRID_COLS - 1, (i32)ceil((x - gridConstraints.minX + searchWindowSize) * gridConstraints.invGridElementWidth));
    if (nMaxCellX < 0)
        return result;

    const i32 nMinCellY = std::max(0, (i32)floor((y - gridConstraints.minY - searchWindowSize) * gridConstraints.invGridElementHeight));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return result;

    const i32 nMaxCellY = std::min((i32)FRAME_GRID_ROWS - 1, (i32)ceil((y - gridConstraints.minY + searchWindowSize) * gridConstraints.invGridElementHeight));
    if (nMaxCellY < 0)
        return result;

    const bool32 checkLevels = (minLevel > 0) || (maxLevel >= 0);

    for (i32 ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (i32 iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = keyPointIndexGrid[ix][iy];

            if (vCell.empty()) continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint& kpUn = undistortedKeyPoints[vCell[j]];
                if (checkLevels)
                {
                    if (kpUn.octave < minLevel) continue;
                    if (maxLevel >= 0 && kpUn.octave > maxLevel) continue;
                }

                const r32 distx = kpUn.pt.x - x;
                const r32 disty = kpUn.pt.y - y;

                if (fabs(distx) < searchWindowSize && fabs(disty) < searchWindowSize)
                {
                    result.push_back(vCell[j]);
                }
            }
        }
    }

    return result;
}

static i32 findInitializationMatches(const std::vector<cv::KeyPoint>& undistortedKeyPoints1,
                                     const std::vector<cv::KeyPoint>& undistortedKeyPoints2,
                                     const cv::Mat&                   descriptors1,
                                     const cv::Mat&                   descriptors2,
                                     const std::vector<size_t>        keyPointIndexGrid2[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                     const std::vector<cv::Point2f>&  previouslyMatchedKeyPoints,
                                     const GridConstraints&           gridConstraints,
                                     const r32                        shortestToSecondShortestDistanceRatio,
                                     const bool32                     checkOrientation,
                                     const i32                        searchWindowSize,
                                     std::vector<i32>&                matches12)
{
    i32 result = 0;
    matches12  = std::vector<i32>(undistortedKeyPoints1.size(), -1);

    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }

    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    std::vector<i32> matchesDistances(undistortedKeyPoints2.size(), INT_MAX);
    std::vector<i32> matches21(undistortedKeyPoints2.size(), -1);

    for (size_t i1 = 0, iend1 = undistortedKeyPoints1.size();
         i1 < iend1;
         i1++)
    {
        cv::KeyPoint keyPoint1 = undistortedKeyPoints1[i1];

        i32 level1 = keyPoint1.octave;
        if (level1 > 0) continue;

        std::vector<size_t> keyPointIndicesCurrentFrame =
          getFeatureIndicesForArea(undistortedKeyPoints2.size(),
                                   searchWindowSize,
                                   previouslyMatchedKeyPoints[i1].x,
                                   previouslyMatchedKeyPoints[i1].y,
                                   gridConstraints,
                                   level1,
                                   level1,
                                   keyPointIndexGrid2,
                                   undistortedKeyPoints2);

        if (keyPointIndicesCurrentFrame.empty()) continue;

        cv::Mat d1 = descriptors1.row(i1);

        // smaller is better
        i32 shortestDist       = INT_MAX;
        i32 secondShortestDist = INT_MAX;
        i32 shortestDistId     = -1;

        for (std::vector<size_t>::iterator vit = keyPointIndicesCurrentFrame.begin();
             vit != keyPointIndicesCurrentFrame.end();
             vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = descriptors2.row(i2);

            i32 dist = descriptorDistance(d1, d2);

            if (matchesDistances[i2] <= dist) continue;

            if (dist < shortestDist)
            {
                secondShortestDist = shortestDist;
                shortestDist       = dist;
                shortestDistId     = i2;
            }
            else if (dist < secondShortestDist)
            {
                secondShortestDist = dist;
            }
        }

        if (shortestDist <= MATCHER_DISTANCE_THRESHOLD_LOW)
        {
            // test that shortest distance is unambiguous
            if (shortestDist < (r32)secondShortestDist * shortestToSecondShortestDistanceRatio)
            {
                // delete previous match, if it exists
                if (matches21[shortestDistId] >= 0)
                {
                    i32 previouslyMatchedKeyPointId        = matches21[shortestDistId];
                    matches12[previouslyMatchedKeyPointId] = -1;
                    result--;
                }

                matches12[i1]                    = shortestDistId;
                matches21[shortestDistId]        = i1;
                matchesDistances[shortestDistId] = shortestDist;
                result++;

                if (checkOrientation)
                {
                    r32 rot = undistortedKeyPoints1[i1].angle - undistortedKeyPoints2[shortestDistId].angle;
                    if (rot < 0.0) rot += 360.0f;

                    i32 bin = round(rot * factor);
                    if (bin == ROTATION_HISTORY_LENGTH) bin = 0;

                    assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);

                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (checkOrientation)
    {
        i32 ind1 = -1;
        i32 ind2 = -1;
        i32 ind3 = -1;

        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

        for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3) continue;

            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                i32 idx1 = rotHist[i][j];
                if (matches12[idx1] >= 0)
                {
                    matches12[idx1] = -1;
                    result--;
                }
            }
        }
    }

    return result;
}

static i32 findMapPointMatchesByBoW(const DBoW2::FeatureVector&      featureVectorReferenceFrame,
                                    const DBoW2::FeatureVector&      featureVectorCurrentFrame,
                                    const std::vector<MapPoint*>&    mapPointMatchesReferenceFrame,
                                    const std::vector<cv::KeyPoint>& undistortedKeyPointsReferenceFrame,
                                    const std::vector<cv::KeyPoint>& keyPointsCurrentFrame,
                                    const cv::Mat&                   descriptorsReferenceFrame,
                                    const cv::Mat&                   descriptorsCurrentFrame,
                                    const bool32                     checkOrientation,
                                    std::vector<MapPoint*>&          matches,
                                    r32                              bestToSecondBestRatio)
{
    i32 result = 0;

    const std::vector<MapPoint*> referenceKeyFrameMapPoints = mapPointMatchesReferenceFrame;
    matches                                                 = std::vector<MapPoint*>(keyPointsCurrentFrame.size(), nullptr);

    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }
    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit  = featureVectorReferenceFrame.begin();
    DBoW2::FeatureVector::const_iterator Fit   = featureVectorCurrentFrame.begin();
    DBoW2::FeatureVector::const_iterator KFend = featureVectorReferenceFrame.end();
    DBoW2::FeatureVector::const_iterator Fend  = featureVectorCurrentFrame.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<u32> vIndicesKF = KFit->second;
            const vector<u32> vIndicesF  = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const u32 realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = referenceKeyFrameMapPoints[realIdxKF];

                if (!pMP) continue;
                if (pMP->bad) continue;

                const cv::Mat& dKF = descriptorsReferenceFrame.row(realIdxKF);

                i32 bestDist1 = 256;
                i32 bestIdxF  = -1;
                i32 bestDist2 = 256;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const u32 realIdxF = vIndicesF[iF];

                    if (matches[realIdxF]) continue;

                    const cv::Mat& dF = descriptorsCurrentFrame.row(realIdxF);

                    const i32 dist = descriptorDistance(dKF, dF);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF  = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 <= MATCHER_DISTANCE_THRESHOLD_LOW)
                {
                    if (static_cast<r32>(bestDist1) < bestToSecondBestRatio * static_cast<r32>(bestDist2))
                    {
                        matches[bestIdxF] = pMP;

                        const cv::KeyPoint& kp = undistortedKeyPointsReferenceFrame[realIdxKF];

                        if (checkOrientation)
                        {
                            // TODO(jan): are we sure that we should not use undistorted keypoints here?
                            r32 rot = kp.angle - keyPointsCurrentFrame[bestIdxF].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            i32 bin = round(rot * factor);
                            if (bin == ROTATION_HISTORY_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        result++;
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = featureVectorReferenceFrame.lower_bound(Fit->first);
        }
        else
        {
            Fit = featureVectorCurrentFrame.lower_bound(KFit->first);
        }
    }

    if (checkOrientation)
    {
        i32 ind1 = -1;
        i32 ind2 = -1;
        i32 ind3 = -1;

        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

        for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3) continue;

            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                matches[rotHist[i][j]] = nullptr;
                result--;
            }
        }
    }

    return result;
}

static r32 getMapPointMaxDistanceInvariance(r32 maxDistance)
{
    r32 result = 1.2f * maxDistance;

    return result;
}

static r32 getMapPointMaxDistanceInvariance(MapPoint* mapPoint)
{
    // TODO(jan): mutex

    r32 result = getMapPointMaxDistanceInvariance(mapPoint->maxDistance);

    return result;
}

static r32 getMapPointMinDistanceInvariance(r32 minDistance)
{
    r32 result = 0.8f * minDistance;

    return result;
}

static r32 getMapPointMinDistanceInvariance(MapPoint* mapPoint)
{
    // TODO(jan): mutex

    r32 result = getMapPointMinDistanceInvariance(mapPoint->minDistance);

    return result;
}

static i32 predictMapPointScale(const r32 maxDistance,
                                const r32 currentDistance,
                                const i32 numberOfScaleLevels,
                                const r32 logScaleFactor)
{
    r32 ratio;
    {
        // TODO(jan): mutex
        ratio = maxDistance / currentDistance;
    }

    i32 result = ceil(log(ratio) / logScaleFactor);
    if (result < 0)
    {
        result = 0;
    }
    else if (result >= numberOfScaleLevels)
    {
        result = numberOfScaleLevels - 1;
    }

    return result;
}

static i32 searchMapPointsByProjection(const cv::Mat&                  cTwCurrentFrame,
                                       const std::vector<size_t>       currentFrameKeyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                       const i32                       currentFrameKeyPointCount,
                                       const std::vector<cv::KeyPoint> currentFrameUndistortedKeyPoints,
                                       const cv::Mat&                  currentFrameDescriptors,
                                       std::vector<MapPoint*>&         currentFrameMapPoints,
                                       const std::vector<MapPoint*>&   candiateKeyFrameMapPoints,
                                       const std::vector<cv::KeyPoint> candiateKeyFrameUndistortedKeyPoints,
                                       const set<MapPoint*>&           alreadyFoundMapPoints,
                                       const GridConstraints           gridConstraints,
                                       const std::vector<r32>&         scaleFactors,
                                       const i32                       numberOfScaleLevels,
                                       const r32                       logScaleFactor,
                                       const r32                       fx,
                                       const r32                       fy,
                                       const r32                       cx,
                                       const r32                       cy,
                                       const r32                       threshold,
                                       const i32                       ORBdist,
                                       const bool32                    checkOrientation)
{
    i32 result = 0;

    const cv::Mat Rcw = cTwCurrentFrame.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = cTwCurrentFrame.rowRange(0, 3).col(3);
    const cv::Mat Ow  = -Rcw.t() * tcw;

    // Rotation Histogram (to check rotation consistency)
    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++) rotHist[i].reserve(500);
    const float factor = 1.0f / ROTATION_HISTORY_LENGTH;

    const vector<MapPoint*> vpMPs = candiateKeyFrameMapPoints;

    for (i32 i = 0; i < vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if (pMP)
        {
            if (!pMP->bad && !alreadyFoundMapPoints.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->position;
                cv::Mat x3Dc = Rcw * x3Dw + tcw;

                const float xc    = x3Dc.at<float>(0);
                const float yc    = x3Dc.at<float>(1);
                const float invzc = 1.0 / x3Dc.at<float>(2);

                const float u = fx * xc * invzc + cx;
                const float v = fy * yc * invzc + cy;

                if (u < gridConstraints.minX || u > gridConstraints.maxX) continue;
                if (v < gridConstraints.minY || v > gridConstraints.maxY) continue;

                // Compute predicted scale level
                cv::Mat PO     = x3Dw - Ow;
                float   dist3D = cv::norm(PO);

                const float maxDistance = getMapPointMaxDistanceInvariance(pMP);
                const float minDistance = getMapPointMinDistanceInvariance(pMP);

                // Depth must be inside the scale pyramid of the image
                if (dist3D < minDistance || dist3D > maxDistance) continue;

                int nPredictedLevel = predictMapPointScale(pMP->maxDistance,
                                                           dist3D,
                                                           numberOfScaleLevels,
                                                           logScaleFactor);

                // Search in a window
                const r32 radius = threshold * scaleFactors[nPredictedLevel];

                const std::vector<size_t> vIndices2 = getFeatureIndicesForArea(currentFrameKeyPointCount,
                                                                               radius,
                                                                               u,
                                                                               v,
                                                                               gridConstraints,
                                                                               nPredictedLevel - 1,
                                                                               nPredictedLevel + 1,
                                                                               currentFrameKeyPointIndexGrid,
                                                                               currentFrameUndistortedKeyPoints);

                if (vIndices2.empty()) continue;

                const cv::Mat dMP = pMP->descriptor;

                int bestDist = 256;
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;

                    if (currentFrameMapPoints[i2]) continue;

                    const cv::Mat& d = currentFrameDescriptors.row(i2);

                    const int dist = descriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= ORBdist)
                {
                    currentFrameMapPoints[bestIdx2] = pMP;
                    result++;

                    if (checkOrientation)
                    {
                        float rot = candiateKeyFrameUndistortedKeyPoints[i].angle - currentFrameUndistortedKeyPoints[bestIdx2].angle;
                        if (rot < 0.0) rot += 360.0f;

                        int bin = round(rot * factor);
                        if (bin == ROTATION_HISTORY_LENGTH) bin = 0;

                        assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if (checkOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < ROTATION_HISTORY_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    currentFrameMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                    result--;
                }
            }
        }
    }

    return result;
}

static bool checkDistEpipolarLine(const cv::KeyPoint&     kp1,
                                  const cv::KeyPoint&     kp2,
                                  const cv::Mat&          F12,
                                  const std::vector<r32>& sigmaSquared)
{ // Epipolar line in second image l = x1'F12 = [a b c]
    const r32 a = kp1.pt.x * F12.at<r32>(0, 0) + kp1.pt.y * F12.at<r32>(1, 0) + F12.at<r32>(2, 0);
    const r32 b = kp1.pt.x * F12.at<r32>(0, 1) + kp1.pt.y * F12.at<r32>(1, 1) + F12.at<r32>(2, 1);
    const r32 c = kp1.pt.x * F12.at<r32>(0, 2) + kp1.pt.y * F12.at<r32>(1, 2) + F12.at<r32>(2, 2);

    const r32 num = a * kp2.pt.x + b * kp2.pt.y + c;

    const r32 den = a * a + b * b;

    if (den == 0) // TODO(jan): this is very bad practice, floating point inprecision
    {
        return false;
    }

    const r32 dsqr = num * num / den;

    return dsqr < 3.84 * sigmaSquared[kp2.octave];
}

static i32 searchMapPointMatchesForTriangulation(KeyFrame*                               keyFrame1,
                                                 KeyFrame*                               keyFrame2,
                                                 r32                                     fx,
                                                 r32                                     fy,
                                                 r32                                     cx,
                                                 r32                                     cy,
                                                 const std::vector<r32>&                 sigmaSquared,
                                                 const std::vector<r32>&                 scaleFactors,
                                                 cv::Mat&                                F12,
                                                 bool32                                  checkOrientation,
                                                 std::vector<std::pair<size_t, size_t>>& vMatchedPairs)
{
    const DBoW2::FeatureVector& vFeatVec1 = keyFrame1->featureVector;
    const DBoW2::FeatureVector& vFeatVec2 = keyFrame2->featureVector;

    //Compute epipole in second image
    cv::Mat Cw  = getKeyFrameCameraCenter(keyFrame1);
    cv::Mat R2w = getKeyFrameRotation(keyFrame2);
    cv::Mat t2w = getKeyFrameTranslation(keyFrame2);
    cv::Mat C2  = R2w * Cw + t2w;

    const r32 invz = 1.0f / C2.at<r32>(2);
    const r32 ex   = fx * C2.at<r32>(0) * invz + cx;
    const r32 ey   = fy * C2.at<r32>(1) * invz + cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    i32                 nmatches = 0;
    std::vector<bool32> vbMatched2(keyFrame2->numberOfKeyPoints, false);
    std::vector<i32>    vMatches12(keyFrame1->numberOfKeyPoints, -1);

    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }

    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = keyFrame1->mapPointMatches[idx1];

                // If there is already a MapPoint skip
                if (pMP1) continue;

                const cv::KeyPoint& kp1 = keyFrame1->undistortedKeyPoints[idx1];
                const cv::Mat&      d1  = keyFrame1->descriptors.row(idx1);

                i32 bestDist = MATCHER_DISTANCE_THRESHOLD_LOW;
                i32 bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = keyFrame2->mapPointMatches[idx2];

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2) continue;

                    const cv::Mat& d2 = keyFrame2->descriptors.row(idx2);

                    const i32 dist = descriptorDistance(d1, d2);

                    if (dist > MATCHER_DISTANCE_THRESHOLD_LOW || dist > bestDist)
                    {
                        continue;
                    }

                    const cv::KeyPoint& kp2 = keyFrame2->undistortedKeyPoints[idx2];

                    //if (!bStereo1 && !bStereo2)
                    //{
                    const r32 distex = ex - kp2.pt.x;
                    const r32 distey = ey - kp2.pt.y;
                    if (distex * distex + distey * distey < 100 * scaleFactors[kp2.octave])
                    {
                        continue;
                    }
                    //}

                    if (checkDistEpipolarLine(kp1, kp2, F12, sigmaSquared))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const cv::KeyPoint& kp2 = keyFrame2->undistortedKeyPoints[bestIdx2];
                    vMatches12[idx1]        = bestIdx2;
                    nmatches++;

                    if (checkOrientation)
                    {
                        r32 rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == ROTATION_HISTORY_LENGTH)
                        {
                            bin = 0;
                        }

                        assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (checkOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < ROTATION_HISTORY_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0) continue;

        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}
