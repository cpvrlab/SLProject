#include <thread>

#include <DUtils/Random.h>

#include <WAIModeOrbSlam2DataOriented.h>
#include "OrbSlamDataOriented/WAIConverter.cpp"
#include "OrbSlamDataOriented/WAIOrbExtraction.cpp"
#include "OrbSlamDataOriented/WAIOrbMatching.cpp"
#include "OrbSlamDataOriented/WAIOrbSlamInitialization.cpp"
#include "OrbSlamDataOriented/WAIPnPSolver.cpp"

WAI::ModeOrbSlam2DataOriented::ModeOrbSlam2DataOriented(SensorCamera* camera, std::string vocabularyPath)
  : Mode(WAI::ModeType_ORB_SLAM2_DATA_ORIENTED), _camera(camera)
{
    _pose = cv::Mat::eye(4, 4, CV_32F);

    r32 scaleFactor          = 1.2f;
    i32 pyramidScaleLevels   = 8;
    i32 numberOfFeatures     = 1000;
    i32 orbPatchSize         = 31;
    i32 orbHalfPatchSize     = 15;
    i32 edgeThreshold        = 19;
    i32 initialFastThreshold = 20;
    i32 minimalFastThreshold = 7;

    _state        = {};
    _state.status = OrbSlamStatus_Initializing;

    _state.maxFramesBetweenKeyFrames = 30;
    _state.minFramesBetweenKeyFrames = 0;

    _state.orbVocabulary    = new ORBVocabulary();
    bool32 vocabularyLoaded = _state.orbVocabulary->loadFromBinaryFile(vocabularyPath);
    if (!vocabularyLoaded)
    {
        printf("Path to ORBVocabulary %s not correct. Could not load vocabulary. Exiting.\n", vocabularyPath.c_str());
        exit(0);
    }

    _state.invertedKeyFrameFile.resize(_state.orbVocabulary->size());

    initializeOrbExtractionParameters(&_state.orbExtractionParameters,
                                      numberOfFeatures,
                                      pyramidScaleLevels,
                                      initialFastThreshold,
                                      minimalFastThreshold,
                                      scaleFactor,
                                      orbPatchSize,
                                      orbHalfPatchSize,
                                      edgeThreshold);
    initializeOrbExtractionParameters(&_state.initializationOrbExtractionParameters,
                                      2 * numberOfFeatures,
                                      pyramidScaleLevels,
                                      initialFastThreshold,
                                      minimalFastThreshold,
                                      scaleFactor,
                                      orbPatchSize,
                                      orbHalfPatchSize,
                                      edgeThreshold);

    _camera->subscribeToUpdate(this);
}

static void computeBoW(const ORBVocabulary*  orbVocabulary,
                       const cv::Mat&        descriptors,
                       DBoW2::BowVector&     bowVector,
                       DBoW2::FeatureVector& featureVector)
{
    if (bowVector.empty() || featureVector.empty())
    {
        std::vector<cv::Mat> currentDescriptor = convertCvMatToDescriptorVector(descriptors);

        orbVocabulary->transform(currentDescriptor, bowVector, featureVector, 4);
    }
}

static void computeBoW(const ORBVocabulary* orbVocabulary,
                       KeyFrame*            keyFrame)
{
    computeBoW(orbVocabulary,
               keyFrame->descriptors,
               keyFrame->bowVector,
               keyFrame->featureVector);
}

static void computeBoW(const ORBVocabulary* orbVocabulary,
                       Frame*               frame)
{
    computeBoW(orbVocabulary,
               frame->descriptors,
               frame->bowVector,
               frame->featureVector);
}

static void updatePoseMatrices(const cv::Mat& cTw,
                               cv::Mat&       cTwKF,
                               cv::Mat&       wTcKF,
                               cv::Mat&       owKF)
{
    cTw.copyTo(cTwKF);

    cv::Mat crw = cTw.rowRange(0, 3).colRange(0, 3);
    cv::Mat ctw = cTw.rowRange(0, 3).col(3);
    cv::Mat wrc = crw.t();

    owKF = -wrc * ctw;

    wTcKF = cv::Mat::eye(4, 4, cTw.type());
    wrc.copyTo(wTcKF.rowRange(0, 3).colRange(0, 3));
    owKF.copyTo(wTcKF.rowRange(0, 3).col(3));
}

static std::vector<KeyFrame*> getBestCovisibilityKeyFrames(const i32                     maxNumberOfKeyFramesToGet,
                                                           const std::vector<KeyFrame*>& orderedConnectedKeyFrames)
{
    std::vector<KeyFrame*> result;

    if (orderedConnectedKeyFrames.size() < maxNumberOfKeyFramesToGet)
    {
        result = orderedConnectedKeyFrames;
    }
    else
    {
        result = std::vector<KeyFrame*>(orderedConnectedKeyFrames.begin(), orderedConnectedKeyFrames.begin() + maxNumberOfKeyFramesToGet);
    }

    return result;
}

static r32 computeSceneMedianDepthForKeyFrame(const KeyFrame* keyFrame)
{
    std::vector<MapPoint*> mapPoints = keyFrame->mapPointMatches;

    std::vector<r32> depths;
    depths.reserve(mapPoints.size());

    cv::Mat crw           = keyFrame->cTw.row(2).colRange(0, 3);
    cv::Mat wrc           = crw.t();
    r32     keyFrameDepth = keyFrame->cTw.at<r32>(2, 3);

    for (i32 i = 0; i < mapPoints.size(); i++)
    {
        if (!mapPoints[i]) continue;

        const MapPoint* mapPoint = mapPoints[i];
        cv::Mat         position = mapPoint->position;
        r32             depth    = wrc.dot(position) + keyFrameDepth;

        depths.push_back(depth);
    }

    std::sort(depths.begin(), depths.end());

    r32 result = depths[(depths.size() - 1) / 2];

    return result;
}

static cv::Mat computeF12(const KeyFrame* keyFrame1,
                          const KeyFrame* keyFrame2,
                          const cv::Mat&  cameraMat)
{
    cv::Mat wr1 = getKeyFrameRotation(keyFrame1);
    cv::Mat wt1 = getKeyFrameTranslation(keyFrame1);
    cv::Mat wr2 = getKeyFrameRotation(keyFrame2);
    cv::Mat wt2 = getKeyFrameTranslation(keyFrame2);

    cv::Mat R12 = wr1 * wr2.t();
    cv::Mat t12 = -wr1 * wr2.t() * wt2 + wt1;

    cv::Mat t12x = (cv::Mat_<float>(3, 3) << 0.0f, -t12.at<float>(2), t12.at<float>(1), t12.at<float>(2), 0.0f, -t12.at<float>(0), -t12.at<float>(1), t12.at<float>(0), 0.0f); //SkewSymmetricMatrix(t12);

    const cv::Mat& K1 = cameraMat;
    const cv::Mat& K2 = cameraMat;

    cv::Mat result = K1.t().inv() * t12x * R12 * K2.inv();
    return result;
}

static void calculateMapPointNormalAndDepth(const cv::Mat&            position,
                                            std::map<KeyFrame*, i32>& observations,
                                            KeyFrame*                 referenceKeyFrame,
                                            const std::vector<r32>    scaleFactors,
                                            const i32                 numberOfScaleLevels,
                                            r32*                      minDistance,
                                            r32*                      maxDistance,
                                            cv::Mat*                  normalVector)
{
    if (observations.empty()) return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    i32     n      = 0;
    for (std::map<KeyFrame*, i32>::iterator it = observations.begin(), itend = observations.end();
         it != itend;
         it++)
    {
        const KeyFrame* keyFrame = it->first;

        cv::Mat Owi     = getKeyFrameCameraCenter(keyFrame);
        cv::Mat normali = position - Owi;
        normal          = normal + normali / cv::norm(normali);
        n++;
    }

    cv::Mat   PC               = position - getKeyFrameCameraCenter(referenceKeyFrame);
    const r32 dist             = cv::norm(PC);
    const i32 level            = referenceKeyFrame->undistortedKeyPoints[observations[referenceKeyFrame]].octave;
    const r32 levelScaleFactor = scaleFactors[level];

    *maxDistance  = dist * levelScaleFactor;
    *minDistance  = *maxDistance / scaleFactors[numberOfScaleLevels - 1];
    *normalVector = normal / n;
}

static bool32 computeBestDescriptorFromObservations(const std::map<KeyFrame*, i32>& observations,
                                                    cv::Mat*                        descriptor)
{
    bool32 result = false;

    // Retrieve all observed descriptors
    std::vector<cv::Mat> descriptors;

    if (!observations.empty())
    {
        descriptors.reserve(observations.size());

        for (std::map<KeyFrame*, i32>::const_iterator mit = observations.begin(), mend = observations.end();
             mit != mend;
             mit++)
        {
            const KeyFrame* keyFrame = mit->first;

            //if (!pKF->isBad())
            descriptors.push_back(keyFrame->descriptors.row(mit->second));
        }

        if (!descriptors.empty())
        {
            // Compute distances between them
            const i32 descriptorsCount = descriptors.size();

            //r32 distances[descriptorsCount][descriptorsCount];
            //we have to allocate dynamically due to a compiler problem with visual studio compiler
            r32** distances = new r32*[descriptorsCount];
            for (size_t i = 0; i < descriptorsCount; ++i)
                distances[i] = new r32[descriptorsCount];

            for (i32 i = 0; i < descriptorsCount; i++)
            {
                distances[i][i] = 0;
                for (i32 j = i + 1; j < descriptorsCount; j++)
                {
                    i32 distij = descriptorDistance(descriptors[i], descriptors[j]);
                    //TODO: why do we store as real though it is an integer value. For sorting also an integer is used
                    distances[i][j] = (r32)distij;
                    distances[j][i] = (r32)distij;
                }
            }

            // Take the descriptor with least median distance to the rest
            i32 bestMedian = INT_MAX;
            i32 bestIndex  = 0;
            for (i32 i = 0; i < descriptorsCount; i++)
            {
                std::vector<i32> sortedDistances(distances[i], distances[i] + descriptorsCount);
                std::sort(sortedDistances.begin(), sortedDistances.end());
                i32 median = sortedDistances[0.5 * (descriptorsCount - 1)];

                if (median < bestMedian)
                {
                    bestMedian = median;
                    bestIndex  = i;
                }
            }

            *descriptor = descriptors[bestIndex].clone();

            result = true;

            //free Distances (remove when vs compiler problem is fixed)
            for (size_t i = 0; i < descriptorsCount; ++i)
                delete distances[i];
            delete distances;
        }
    }

    return result;
}

static void updateKeyFrameOrderedCovisibilityVectors(std::map<KeyFrame*, i32>& connectedKeyFrameWeights,
                                                     std::vector<KeyFrame*>&   orderedConnectedKeyFrames,
                                                     std::vector<i32>&         orderedWeights)
{
    std::vector<std::pair<i32, KeyFrame*>> vPairs;

    vPairs.reserve(connectedKeyFrameWeights.size());
    for (std::map<KeyFrame*, i32>::iterator mit = connectedKeyFrameWeights.begin(), mend = connectedKeyFrameWeights.end();
         mit != mend;
         mit++)
    {
        vPairs.push_back(std::make_pair(mit->second, mit->first));
    }

    std::sort(vPairs.begin(), vPairs.end());

    std::list<KeyFrame*> connectedKeyFrames;
    std::list<i32>       weights;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        connectedKeyFrames.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    orderedConnectedKeyFrames = std::vector<KeyFrame*>(connectedKeyFrames.begin(), connectedKeyFrames.end());
    orderedWeights            = std::vector<i32>(weights.begin(), weights.end());
}

static void addKeyFrameCovisibilityConnection(KeyFrame*                 keyFrame,
                                              const i32                 weight,
                                              std::map<KeyFrame*, i32>& connectedKeyFrameWeights,
                                              std::vector<KeyFrame*>&   orderedConnectedKeyFrames,
                                              std::vector<i32>&         orderedWeights)
{
    if (!connectedKeyFrameWeights.count(keyFrame))
    {
        connectedKeyFrameWeights[keyFrame] = weight;
    }
    else if (connectedKeyFrameWeights[keyFrame] != weight)
    {
        connectedKeyFrameWeights[keyFrame] = weight;
    }
    else
    {
        return;
    }

    updateKeyFrameOrderedCovisibilityVectors(connectedKeyFrameWeights, orderedConnectedKeyFrames, orderedWeights);
}

static void updateKeyFrameConnections(KeyFrame* keyFrame)
{
    std::map<KeyFrame*, i32> keyFrameCounter; // first is the index of the keyframe in keyframes, second is the number of common mapPoints

    //For all map points in keyframe check in which other keyframes are they seeing
    //Increase counter for those keyframes
    for (std::vector<MapPoint*>::const_iterator vit = keyFrame->mapPointMatches.begin(), vend = keyFrame->mapPointMatches.end();
         vit != vend;
         vit++)
    {
        MapPoint* mapPoint = *vit;

        if (!mapPoint) continue;
        if (mapPoint->bad) continue;

        std::map<KeyFrame*, i32> observations = mapPoint->observations;

        for (std::map<KeyFrame*, i32>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend;
             mit++)
        {
            if (mit->first == keyFrame) continue;

            keyFrameCounter[mit->first]++;
        }
    }

    // This should not happen
    if (keyFrameCounter.empty()) return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    i32       maxCommonMapPointCount      = 0;
    KeyFrame* keyFrameWithMaxCommonPoints = nullptr;
    i32       threshold                   = 15;

    std::vector<std::pair<i32, KeyFrame*>> vPairs;
    vPairs.reserve(keyFrameCounter.size());
    for (std::map<KeyFrame*, i32>::iterator mit = keyFrameCounter.begin(), mend = keyFrameCounter.end();
         mit != mend;
         mit++)
    {
        KeyFrame* connectedKeyFrame = mit->first;
        i32       weight            = mit->second;

        if (weight > maxCommonMapPointCount)
        {
            maxCommonMapPointCount      = weight;
            keyFrameWithMaxCommonPoints = connectedKeyFrame;
        }
        if (weight >= threshold)
        {
            vPairs.push_back(std::make_pair(weight, connectedKeyFrame));

            addKeyFrameCovisibilityConnection(keyFrame,
                                              weight,
                                              connectedKeyFrame->connectedKeyFrameWeights,
                                              connectedKeyFrame->orderedConnectedKeyFrames,
                                              connectedKeyFrame->orderedWeights);
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(std::make_pair(maxCommonMapPointCount, keyFrameWithMaxCommonPoints));

        KeyFrame* connectedKeyFrame = keyFrameWithMaxCommonPoints;

        addKeyFrameCovisibilityConnection(keyFrame,
                                          maxCommonMapPointCount,
                                          connectedKeyFrame->connectedKeyFrameWeights,
                                          connectedKeyFrame->orderedConnectedKeyFrames,
                                          connectedKeyFrame->orderedWeights);
    }

    sort(vPairs.begin(), vPairs.end());

    std::list<KeyFrame*> connectedKeyFrames;
    std::list<i32>       weights;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        connectedKeyFrames.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    keyFrame->connectedKeyFrameWeights  = keyFrameCounter;
    keyFrame->orderedConnectedKeyFrames = std::vector<KeyFrame*>(connectedKeyFrames.begin(), connectedKeyFrames.end());
    keyFrame->orderedWeights            = std::vector<i32>(weights.begin(), weights.end());

    // TODO(jan): add child and parent connections
#if 0
    if (mbFirstConnection && mnId != 0)
    {
        mpParent = mvpOrderedConnectedKeyFrames.front();
        mpParent->AddChild(this);
        mbFirstConnection = false;
    }
#endif
}

static i32 createNewMapPoints(KeyFrame*                     keyFrame,
                              const std::vector<KeyFrame*>& orderedConnectedKeyFrames,
                              const r32                     fx,
                              const r32                     fy,
                              const r32                     cx,
                              const r32                     cy,
                              const r32                     invfx,
                              const r32                     invfy,
                              const i32                     numberOfScaleLevels,
                              const r32                     scaleFactor,
                              const cv::Mat&                cameraMat,
                              const std::vector<r32>&       sigmaSquared,
                              const std::vector<r32>&       scaleFactors,
                              std::set<MapPoint*>&          mapPoints,
                              i32&                          nextMapPointIndex,
                              std::list<MapPoint*>&         newMapPoints)
{
    i32 result = 0;

    const std::vector<KeyFrame*> neighboringKeyFrames = getBestCovisibilityKeyFrames(20,
                                                                                     orderedConnectedKeyFrames);

    cv::Mat crw1 = getKeyFrameRotation(keyFrame);
    cv::Mat wrc1 = crw1.t();
    cv::Mat ctw1 = getKeyFrameTranslation(keyFrame);
    cv::Mat cTw1(3, 4, CV_32F);

    crw1.copyTo(cTw1.colRange(0, 3));
    ctw1.copyTo(cTw1.col(3));
    cv::Mat origin1 = getKeyFrameCameraCenter(keyFrame);

    const r32& fx1    = fx;
    const r32& fy1    = fy;
    const r32& cx1    = cx;
    const r32& cy1    = cy;
    const r32& invfx1 = invfx;
    const r32& invfy1 = invfy;

    const r32 ratioFactor = 1.5f * scaleFactor;

    // Search matches with epipolar restriction and triangulate
    for (i32 i = 0; i < neighboringKeyFrames.size(); i++)
    {
        // TODO(jan): reactivate
        //if (i > 0 && CheckNewKeyFrames()) return;

        KeyFrame* neighboringKeyFrame = neighboringKeyFrames[i];

        // Check first that baseline is not too short
        cv::Mat origin2   = getKeyFrameCameraCenter(neighboringKeyFrame);
        cv::Mat vBaseline = origin2 - origin1;

        const r32 baseline = cv::norm(vBaseline);

        const r32 medianDepthNeighboringKeyFrame = computeSceneMedianDepthForKeyFrame(neighboringKeyFrame);
        const r32 ratioBaselineDepth             = baseline / medianDepthNeighboringKeyFrame;

        if (ratioBaselineDepth < 0.01) continue;

        // Compute Fundamental Matrix
        cv::Mat F12 = computeF12(keyFrame, neighboringKeyFrame, cameraMat);

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t, size_t>> matchedIndices;
        searchMapPointMatchesForTriangulation(keyFrame, neighboringKeyFrame, fx, fy, cx, cy, sigmaSquared, scaleFactors, F12, false, matchedIndices);

        cv::Mat crw2 = getKeyFrameRotation(neighboringKeyFrame);
        cv::Mat wrc2 = crw2.t();
        cv::Mat ctw2 = getKeyFrameTranslation(neighboringKeyFrame);
        cv::Mat cTw2(3, 4, CV_32F);
        crw2.copyTo(cTw2.colRange(0, 3));
        ctw2.copyTo(cTw2.col(3));

        const r32& fx2    = fx;
        const r32& fy2    = fy;
        const r32& cx2    = cx;
        const r32& cy2    = cy;
        const r32& invfx2 = invfx;
        const r32& invfy2 = invfy;

        // Triangulate each match
        const i32 matchCount = matchedIndices.size();
        for (i32 matchIndex = 0; matchIndex < matchCount; matchIndex++)
        {
            const i32& keyPointIndex1 = matchedIndices[matchIndex].first;
            const i32& keyPointIndex2 = matchedIndices[matchIndex].second;

            const cv::KeyPoint& kp1 = keyFrame->undistortedKeyPoints[keyPointIndex1];
            const cv::KeyPoint& kp2 = neighboringKeyFrame->undistortedKeyPoints[keyPointIndex2];

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<r32>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<r32>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat   ray1            = wrc1 * xn1;
            cv::Mat   ray2            = wrc2 * xn2;
            const r32 cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            r32 cosParallaxStereo = cosParallaxRays + 1;

            cv::Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && cosParallaxRays < 0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<r32>(0) * cTw1.row(2) - cTw1.row(0);
                A.row(1) = xn1.at<r32>(1) * cTw1.row(2) - cTw1.row(1);
                A.row(2) = xn2.at<r32>(0) * cTw2.row(2) - cTw2.row(0);
                A.row(3) = xn2.at<r32>(1) * cTw2.row(2) - cTw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<r32>(3) == 0) continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<r32>(3);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            r32 z1 = crw1.row(2).dot(x3Dt) + ctw1.at<r32>(2);
            if (z1 <= 0) continue;

            r32 z2 = crw2.row(2).dot(x3Dt) + ctw2.at<r32>(2);
            if (z2 <= 0) continue;

            //Check reprojection error in first keyframe
            const r32& sigmaSquare1 = sigmaSquared[kp1.octave];
            const r32  x1           = crw1.row(0).dot(x3Dt) + ctw1.at<r32>(0);
            const r32  y1           = crw1.row(1).dot(x3Dt) + ctw1.at<r32>(1);
            const r32  invz1        = 1.0 / z1;

            r32 u1    = fx1 * x1 * invz1 + cx1;
            r32 v1    = fy1 * y1 * invz1 + cy1;
            r32 errX1 = u1 - kp1.pt.x;
            r32 errY1 = v1 - kp1.pt.y;

            if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) continue;

            //Check reprojection error in second keyframe
            const r32 sigmaSquare2 = sigmaSquared[kp2.octave];
            const r32 x2           = crw2.row(0).dot(x3Dt) + ctw2.at<r32>(0);
            const r32 y2           = crw2.row(1).dot(x3Dt) + ctw2.at<r32>(1);
            const r32 invz2        = 1.0 / z2;

            r32 u2    = fx2 * x2 * invz2 + cx2;
            r32 v2    = fy2 * y2 * invz2 + cy2;
            r32 errX2 = u2 - kp2.pt.x;
            r32 errY2 = v2 - kp2.pt.y;

            if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) continue;

            //Check scale consistency
            cv::Mat normal1 = x3D - origin1;
            r32     dist1   = cv::norm(normal1);

            cv::Mat normal2 = x3D - origin2;
            r32     dist2   = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0) continue;

            const r32 ratioDist   = dist2 / dist1;
            const r32 ratioOctave = scaleFactors[kp1.octave] / scaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) continue;

            // Triangulation is succesful
            MapPoint* mapPoint = new MapPoint();
            initializeMapPoint(mapPoint, keyFrame, x3D, nextMapPointIndex);

            keyFrame->mapPointMatches[keyPointIndex1]            = mapPoint;
            neighboringKeyFrame->mapPointMatches[keyPointIndex2] = mapPoint;

            mapPoint->observations[keyFrame]            = keyPointIndex1;
            mapPoint->observations[neighboringKeyFrame] = keyPointIndex2;

            computeBestDescriptorFromObservations(mapPoint->observations,
                                                  &mapPoint->descriptor);
            calculateMapPointNormalAndDepth(mapPoint->position,
                                            mapPoint->observations,
                                            keyFrame,
                                            scaleFactors,
                                            numberOfScaleLevels,
                                            &mapPoint->minDistance,
                                            &mapPoint->maxDistance,
                                            &mapPoint->normalVector);

            mapPoints.insert(mapPoint);
            newMapPoints.push_back(mapPoint);
            result++;
        }
    }

    return result;
}

static void initializeKeyFrame(KeyFrame*    keyFrame,
                               const Frame& referenceFrame,
                               i32&         nextKeyFrameId)
{
    *keyFrame = {};

    keyFrame->index = nextKeyFrameId++;

    keyFrame->frameId              = referenceFrame.id;
    keyFrame->numberOfKeyPoints    = referenceFrame.numberOfKeyPoints;
    keyFrame->keyPoints            = referenceFrame.keyPoints;
    keyFrame->undistortedKeyPoints = referenceFrame.undistortedKeyPoints;
    keyFrame->descriptors          = referenceFrame.descriptors.clone();
    keyFrame->bowVector            = referenceFrame.bowVector;
    keyFrame->featureVector        = referenceFrame.featureVector;
    keyFrame->cameraMat            = referenceFrame.cameraMat.clone();
    keyFrame->mapPointMatches      = referenceFrame.mapPointMatches;

    for (i32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (i32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyFrame->keyPointIndexGrid[i][j] = referenceFrame.keyPointIndexGrid[i][j];
        }
    }

    updatePoseMatrices(referenceFrame.cTw,
                       keyFrame->cTw,
                       keyFrame->wTc,
                       keyFrame->worldOrigin);
}

static void initializeFrame(Frame*                         frame,
                            i32&                           nextFrameId,
                            const cv::Mat&                 cameraFrame,
                            const cv::Mat&                 cameraMat,
                            const cv::Mat&                 distortionMat,
                            const OrbExtractionParameters& orbExtractionParameters,
                            const GridConstraints          gridConstraints)
{
    *frame = {};

    frame->id = nextFrameId++;

    frame->cameraMat = cameraMat.clone();

    std::vector<cv::Mat> imagePyramid;
    imagePyramid.resize(orbExtractionParameters.numberOfScaleLevels);

    // Compute scaled images according to scale factors
    computeScalePyramid(cameraFrame,
                        orbExtractionParameters.numberOfScaleLevels,
                        orbExtractionParameters.inverseScaleFactors,
                        orbExtractionParameters.edgeThreshold,
                        imagePyramid);

    // Compute key points, distributed in an evenly spaced grid
    // on every scale level
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
    computeKeyPointsInOctTree(imagePyramid,
                              orbExtractionParameters.numberOfScaleLevels,
                              orbExtractionParameters.scaleFactors,
                              orbExtractionParameters.edgeThreshold,
                              orbExtractionParameters.numberOfFeatures,
                              orbExtractionParameters.numberOfFeaturesPerScaleLevel,
                              orbExtractionParameters.initialThreshold,
                              orbExtractionParameters.minimalThreshold,
                              orbExtractionParameters.orbOctTreePatchSize,
                              orbExtractionParameters.orbOctTreeHalfPatchSize,
                              orbExtractionParameters.umax,
                              allKeyPoints);

    i32 nkeypoints = 0;
    for (i32 level = 0; level < orbExtractionParameters.numberOfScaleLevels; ++level)
    {
        nkeypoints += (i32)allKeyPoints[level].size();
    }

    cv::Mat descriptors;
    if (nkeypoints == 0)
    {
        frame->descriptors.release();
    }
    else
    {
        frame->descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = frame->descriptors;
    }

    frame->keyPoints.clear();
    frame->keyPoints.reserve(nkeypoints);

    i32 offset = 0;
    for (i32 level = 0; level < orbExtractionParameters.numberOfScaleLevels; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints       = allKeyPoints[level];
        i32                        nkeypointsLevel = (i32)keypoints.size();

        if (nkeypointsLevel == 0) continue;

        // preprocess the resized image
        cv::Mat workingMat = imagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // Compute the descriptors
        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        desc         = cv::Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

        for (size_t i = 0; i < keypoints.size(); i++)
        {
            computeOrbDescriptor(keypoints[i], workingMat, &orbExtractionParameters.orbPattern[0], desc.ptr((i32)i));
        }

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            r32 scale = orbExtractionParameters.scaleFactors[level];
            for (std::vector<cv::KeyPoint>::iterator keypoint    = keypoints.begin(),
                                                     keypointEnd = keypoints.end();
                 keypoint != keypointEnd;
                 ++keypoint)
            {
                keypoint->pt *= scale;
            }
        }
        // And add the keypoints to the output
        frame->keyPoints.insert(frame->keyPoints.end(), keypoints.begin(), keypoints.end());
    }

    frame->numberOfKeyPoints = frame->keyPoints.size();

    if (!frame->keyPoints.empty())
    {
        undistortKeyPoints(cameraMat,
                           distortionMat,
                           frame->keyPoints,
                           frame->numberOfKeyPoints,
                           frame->undistortedKeyPoints);

        frame->mapPointMatches   = std::vector<MapPoint*>(frame->numberOfKeyPoints, static_cast<MapPoint*>(NULL));
        frame->mapPointIsOutlier = std::vector<bool32>(frame->numberOfKeyPoints, false);

        i32 nReserve = 0.5f * frame->numberOfKeyPoints / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (u32 i = 0; i < FRAME_GRID_COLS; i++)
        {
            for (u32 j = 0; j < FRAME_GRID_ROWS; j++)
            {
                frame->keyPointIndexGrid[i][j].reserve(nReserve);
            }
        }

        for (i32 i = 0; i < frame->numberOfKeyPoints; i++)
        {
            const cv::KeyPoint& kp = frame->undistortedKeyPoints[i];

            i32    xPos, yPos;
            bool32 keyPointIsInGrid = calculateKeyPointGridCell(kp, gridConstraints, &xPos, &yPos);
            if (keyPointIsInGrid)
            {
                frame->keyPointIndexGrid[xPos][yPos].push_back(i);
            }
        }

        // TODO(jan): 'retain image' functionality
    }
}

static void initializeFrame(Frame*       frame,
                            const Frame& referenceFrame)
{
    *frame = {};

    frame->id                   = referenceFrame.id;
    frame->cameraMat            = referenceFrame.cameraMat.clone();
    frame->numberOfKeyPoints    = referenceFrame.numberOfKeyPoints;
    frame->keyPoints            = referenceFrame.keyPoints;
    frame->undistortedKeyPoints = referenceFrame.undistortedKeyPoints;
    frame->descriptors          = referenceFrame.descriptors.clone();
    frame->mapPointMatches      = referenceFrame.mapPointMatches;
    frame->mapPointIsOutlier    = referenceFrame.mapPointIsOutlier;
    frame->referenceKeyFrame    = referenceFrame.referenceKeyFrame;
    frame->bowVector            = referenceFrame.bowVector;
    frame->featureVector        = referenceFrame.featureVector;

    for (i32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (i32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            frame->keyPointIndexGrid[i][j] = referenceFrame.keyPointIndexGrid[i][j];
        }
    }

    if (!referenceFrame.cTw.empty())
    {
        updatePoseMatrices(referenceFrame.cTw,
                           frame->cTw,
                           frame->wTc,
                           frame->worldOrigin);
    }
}

void computeGridConstraints(const cv::Mat&   cameraFrame,
                            const cv::Mat&   cameraMat,
                            const cv::Mat&   distortionMat,
                            GridConstraints* gridConstraints)
{
    r32 minX, maxX, minY, maxY;

    if (distortionMat.at<r32>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<r32>(0, 0) = 0.0;
        mat.at<r32>(0, 1) = 0.0;
        mat.at<r32>(1, 0) = cameraFrame.cols;
        mat.at<r32>(1, 1) = 0.0;
        mat.at<r32>(2, 0) = 0.0;
        mat.at<r32>(2, 1) = cameraFrame.rows;
        mat.at<r32>(3, 0) = cameraFrame.cols;
        mat.at<r32>(3, 1) = cameraFrame.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, cameraMat, distortionMat, cv::Mat(), cameraMat);
        mat = mat.reshape(1);

        minX = (r32)std::min(mat.at<r32>(0, 0), mat.at<r32>(2, 0));
        maxX = (r32)std::max(mat.at<r32>(1, 0), mat.at<r32>(3, 0));
        minY = (r32)std::min(mat.at<r32>(0, 1), mat.at<r32>(1, 1));
        maxY = (r32)std::max(mat.at<r32>(2, 1), mat.at<r32>(3, 1));
    }
    else
    {
        minX = 0.0f;
        maxX = cameraFrame.cols;
        minY = 0.0f;
        maxY = cameraFrame.rows;
    }

    gridConstraints->minX                 = minX;
    gridConstraints->minY                 = minY;
    gridConstraints->maxX                 = maxX;
    gridConstraints->maxY                 = maxY;
    gridConstraints->invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(maxX - minX);
    gridConstraints->invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(maxY - minY);
}

i32 countMapPointsObservedByKeyFrame(const KeyFrame* keyFrame,
                                     const i32       minObservationsCount)
{
    i32 result = 0;

    const bool32 checkObservations = minObservationsCount > 0;

    if (checkObservations)
    {
        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (!mapPoint->bad)
                {
                    if (mapPoint->observations.size() >= minObservationsCount)
                    {
                        result++;
                    }
                }
            }
        }
    }
    else
    {
        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (!mapPoint->bad)
                {
                    result++;
                }
            }
        }
    }

    return result;
}

static i32 searchMapPointsByProjectionOfCandidateKeyFrameMapPoints(const cv::Mat&                  cTwCurrentFrame,
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

i32 searchMapPointsByProjectionOfLocalMapPoints(std::vector<MapPoint*>& localMapPoints,
                                                const std::vector<r32>& scaleFactors,
                                                const GridConstraints&  gridConstraints,
                                                const i32               thresholdHigh,
                                                const r32               bestToSecondBestRatio,
                                                Frame*                  frame,
                                                i32                     threshold)
{
    i32 result = 0;

    const bool32 factor = threshold != 1.0;

    for (i32 i = 0; i < localMapPoints.size(); i++)
    {
        MapPoint*                    mapPoint     = localMapPoints[i];
        const MapPointTrackingInfos* trackingInfo = &mapPoint->trackingInfos;

        if (!trackingInfo->inView) continue;
        if (mapPoint->bad) continue;

        const i32& predictedLevel = trackingInfo->scaleLevel;

        // The size of the window will depend on the viewing direction
        r32 r = (trackingInfo->viewCos > 0.998f) ? 2.5f : 4.0f;

        if (factor)
        {
            r *= (r32)threshold;
        }

        std::vector<size_t> indices =
          getFeatureIndicesForArea(frame->numberOfKeyPoints,
                                   r * scaleFactors[predictedLevel],
                                   trackingInfo->projX,
                                   trackingInfo->projY,
                                   gridConstraints,
                                   predictedLevel - 1,
                                   predictedLevel,
                                   frame->keyPointIndexGrid,
                                   frame->undistortedKeyPoints);

        if (indices.empty()) continue;

        const cv::Mat descriptor1 = mapPoint->descriptor.clone();

        i32 bestDist   = 256;
        i32 bestLevel  = -1;
        i32 bestDist2  = 256;
        i32 bestLevel2 = -1;
        i32 bestIdx    = -1;

        // Get best and second matches with near keypoints
        for (std::vector<size_t>::const_iterator vit = indices.begin(), vend = indices.end();
             vit != vend;
             vit++)
        {
            const i32 idx      = *vit;
            MapPoint* mapPoint = frame->mapPointMatches[idx];

            if (mapPoint && mapPoint->observations.size() > 0) continue;

            const cv::Mat& descriptor2 = frame->descriptors.row(idx);

            const i32 dist = descriptorDistance(descriptor1, descriptor2);

            if (dist < bestDist)
            {
                bestDist2  = bestDist;
                bestDist   = dist;
                bestLevel2 = bestLevel;
                bestLevel  = frame->undistortedKeyPoints[idx].octave;
                bestIdx    = idx;
            }
            else if (dist < bestDist2)
            {
                bestLevel2 = frame->undistortedKeyPoints[idx].octave;
                bestDist2  = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= thresholdHigh)
        {
            if (bestLevel == bestLevel2 && bestDist > bestToSecondBestRatio * bestDist2) continue;

            frame->mapPointMatches[bestIdx] = mapPoint;
            result++;
        }
    }

    return result;
}

static bool32 needNewKeyFrame(const i32       currentFrameId,
                              const i32       lastKeyFrameId,
                              const i32       lastRelocalizationFrameId,
                              const i32       minFramesBetweenKeyFrames,
                              const i32       maxFramesBetweenKeyFrames,
                              const i32       mapPointMatchCount,
                              const KeyFrame* referenceKeyFrame,
                              const i32       keyFrameCount)
{
    // TODO(jan): check if local mapper is stopped

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (currentFrameId < lastRelocalizationFrameId + maxFramesBetweenKeyFrames &&
        keyFrameCount > maxFramesBetweenKeyFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    i32 minObservationsCount = 3;
    if (keyFrameCount <= 2)
    {
        minObservationsCount = 2;
    }

    i32 referenceMatchCount = countMapPointsObservedByKeyFrame(referenceKeyFrame, minObservationsCount);

// Local Mapping accept keyframes?
// TODO(jan): local mapping
#if 0
    bool32 localMappingIsIdle = mpLocalMapper->AcceptKeyFrames();
#else
    bool32 localMappingIsIdle = true;
#endif

    // Thresholds
    r32 thRefRatio = 0.9f;
    if (keyFrameCount < 2)
    {
        thRefRatio = 0.4f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool32 c1a = currentFrameId >= lastKeyFrameId + maxFramesBetweenKeyFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool32 c1b = (currentFrameId >= lastKeyFrameId + minFramesBetweenKeyFrames && localMappingIsIdle);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool32 c2 = ((mapPointMatchCount < referenceMatchCount * thRefRatio) && mapPointMatchCount > 15);

    if ((c1a || c1b) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (localMappingIsIdle)
        {
            WAI_LOG("[WAITrackedMapping] NeedNewKeyFrame: YES bLocalMappingIdle!");
            return true;
        }
        else
        {
            // TODO(jan): local mapping
            //mpLocalMapper->InterruptBA();
            WAI_LOG("[WAITrackedMapping] NeedNewKeyFrame: NO InterruptBA!");
            return false;
        }
    }
    else
    {
        WAI_LOG("NeedNewKeyFrame: NO!");
        return false;
    }
}

static i32 predictMapPointScale(const r32& currentDist,
                                const r32  maxDistance,
                                const r32  frameScaleFactor,
                                const r32  numberOfScaleLevels)
{
    r32 ratio = maxDistance / currentDist;

    i32 result = ceil(log(ratio) / frameScaleFactor);

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

static bool32 isMapPointInFrameFrustum(const Frame*           frame,
                                       const r32              fx,
                                       const r32              fy,
                                       const r32              cx,
                                       const r32              cy,
                                       const r32              minX,
                                       const r32              maxX,
                                       const r32              minY,
                                       const r32              maxY,
                                       MapPoint*              mapPoint,
                                       r32                    viewingCosLimit,
                                       MapPointTrackingInfos* trackingInfos)
{
    //pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = mapPoint->position;

    // 3D in camera coordinates
    cv::Mat       crw = frame->cTw.rowRange(0, 3).colRange(0, 3); // mRcw in WAIFrame
    cv::Mat       ctw = frame->cTw.rowRange(0, 3).col(3);         // mtcw in WAIFrame
    const cv::Mat Pc  = crw * P + ctw;
    const r32&    PcX = Pc.at<r32>(0);
    const r32&    PcY = Pc.at<r32>(1);
    const r32&    PcZ = Pc.at<r32>(2);

    // Check positive depth
    if (PcZ < 0.0f) return false;

    // Project in image and check it is not outside
    const r32 invz = 1.0f / PcZ;
    const r32 u    = fx * PcX * invz + cx;
    const r32 v    = fy * PcY * invz + cy;

    if (u < minX || u > maxX) return false;
    if (v < minY || v > maxY) return false;

    // Check distance is in the scale invariance region of the WAIMapPoint
    // TODO(jan): magic numbers
    const r32     maxDistance = 1.2f * mapPoint->maxDistance;
    const r32     minDistance = 0.8f * mapPoint->minDistance;
    const cv::Mat PO          = P - (-crw.t() * ctw); // mOw in WAIFrame
    const r32     dist        = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance) return false;

    // Check viewing angle
    cv::Mat Pn = mapPoint->normalVector;

    const r32 viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit) return false;

    // Predict scale in the image
    // TODO(jan): magic numbers
    const i32 predictedLevel = predictMapPointScale(dist, maxDistance, 1.2f, 8);

    // Data used by the tracking
    trackingInfos->inView     = true;
    trackingInfos->projX      = u;
    trackingInfos->projY      = v;
    trackingInfos->scaleLevel = predictedLevel;
    trackingInfos->viewCos    = viewCos;

    return true;
}

// pose otimization
static i32 optimizePose(const std::vector<r32> inverseSigmaSquared,
                        const r32              fx,
                        const r32              fy,
                        const r32              cx,
                        const r32              cy,
                        Frame*                 frame)
{
    i32 result = 0;

    //ghm1: Attention, we add every map point associated to a keypoint to the optimizer
    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    i32 initialCorrespondenceCount = 0;

    g2o::SE3Quat quat = convertCvMatToG2OSE3Quat(frame->cTw);

    // Set Frame vertex
    g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
    vertex->setEstimate(quat);
    vertex->setId(0);
    vertex->setFixed(false);
    optimizer.addVertex(vertex);

    // Set WAIMapPoint vertices
    const i32 keyPointCount = frame->numberOfKeyPoints;

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    std::vector<size_t>                          edgeIndices;
    edges.reserve(keyPointCount);
    edgeIndices.reserve(keyPointCount);

    const r32 kernelDelta = sqrt(5.991);

    {
        for (i32 i = 0; i < frame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = frame->mapPointMatches[i];

            if (mapPoint)
            {
                initialCorrespondenceCount++;
                frame->mapPointIsOutlier[i] = false;

                Eigen::Matrix<r64, 2, 1> observationMatrix;
                const cv::KeyPoint&      undistortKeyPoint = frame->undistortedKeyPoints[i];
                observationMatrix << undistortKeyPoint.pt.x, undistortKeyPoint.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                edge->setMeasurement(observationMatrix);
                const r32 invSigmaSquared = inverseSigmaSquared[undistortKeyPoint.octave];
                edge->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquared);

                g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber;
                edge->setRobustKernel(kernel);
                kernel->setDelta(kernelDelta);

                edge->fx    = fx;
                edge->fy    = fy;
                edge->cx    = cx;
                edge->cy    = cy;
                cv::Mat Xw  = mapPoint->position;
                edge->Xw[0] = Xw.at<r32>(0);
                edge->Xw[1] = Xw.at<r32>(1);
                edge->Xw[2] = Xw.at<r32>(2);

                optimizer.addEdge(edge);

                edges.push_back(edge);
                edgeIndices.push_back(i);
            }
        }
    }

    if (initialCorrespondenceCount >= 3)
    {
        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const r32 chiSquared[4]      = {5.991, 5.991, 5.991, 5.991};
        const i32 iterationCounts[4] = {10, 10, 10, 10};

        i32 badMapPointCount = 0;
        for (i32 iteration = 0; iteration < 4; iteration++)
        {
            quat = convertCvMatToG2OSE3Quat(frame->cTw);
            vertex->setEstimate(quat);
            optimizer.initializeOptimization(0);
            optimizer.optimize(iterationCounts[iteration]);

            badMapPointCount = 0;
            for (i32 i = 0, iend = edges.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* edge = edges[i];

                const i32 edgeIndex = edgeIndices[i];

                if (frame->mapPointIsOutlier[edgeIndex])
                {
                    edge->computeError();
                }

                const r32 chi2 = edge->chi2();

                if (chi2 > chiSquared[iteration])
                {
                    frame->mapPointIsOutlier[edgeIndex] = true;
                    edge->setLevel(1);
                    badMapPointCount++;
                }
                else
                {
                    frame->mapPointIsOutlier[edgeIndex] = false;
                    edge->setLevel(0);
                }

                if (iteration == 2)
                {
                    edge->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10) break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat          quat   = vertex->estimate();
        cv::Mat               pose   = convertG2OSE3QuatToCvMat(quat);

        updatePoseMatrices(pose,
                           frame->cTw,
                           frame->wTc,
                           frame->worldOrigin);
        result = initialCorrespondenceCount - badMapPointCount;
    }

    return result;
}

bool32 computeGeometricalModel(const std::vector<cv::KeyPoint>& undistortedKeyPoints1,
                               const std::vector<cv::KeyPoint>& undistortedKeyPoints2,
                               const cv::Mat&                   descriptors1,
                               const cv::Mat&                   descriptors2,
                               const std::vector<i32>&          initializationMatches,
                               const cv::Mat&                   cameraMat,
                               cv::Mat&                         rcw,
                               cv::Mat&                         tcw,
                               std::vector<bool32>&             keyPointTriangulatedFlags,
                               std::vector<cv::Point3f>&        initialPoints)
{
    bool32 result = false;

    const i32 maxRansacIterations = 200;
    const r32 sigma               = 1.0f;

    std::vector<Match>  matches;
    std::vector<bool32> matched;

    matches.reserve(undistortedKeyPoints2.size());
    matched.resize(undistortedKeyPoints1.size());
    for (size_t i = 0, iend = initializationMatches.size(); i < iend; i++)
    {
        if (initializationMatches[i] >= 0)
        {
            matches.push_back(std::make_pair(i, initializationMatches[i]));
            matched[i] = true;
        }
        else
        {
            matched[i] = false;
        }
    }

    const i32 N = matches.size();

    // Indices for minimum set selection
    std::vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    std::vector<size_t> vAvailableIndices;

    for (i32 i = 0; i < N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    std::vector<std::vector<size_t>> ransacSets = std::vector<std::vector<size_t>>(maxRansacIterations, std::vector<size_t>(8, 0));

    DUtils::Random::SeedRandOnce(1337);

    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            i32 randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            i32 idx   = vAvailableIndices[randi];

            ransacSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    std::vector<bool32> vbMatchesInliersH, vbMatchesInliersF;
    r32                 scoreHomography, scoreFundamental;
    cv::Mat             homography, fundamental;

    std::thread threadHomography(&findHomography,
                                 std::ref(matches),
                                 std::ref(undistortedKeyPoints1),
                                 std::ref(undistortedKeyPoints2),
                                 maxRansacIterations,
                                 std::ref(ransacSets),
                                 sigma,
                                 std::ref(scoreHomography),
                                 std::ref(vbMatchesInliersH),
                                 std ::ref(homography));
    std::thread threadFundamental(&findFundamental,
                                  std::ref(matches),
                                  std::ref(undistortedKeyPoints1),
                                  std::ref(undistortedKeyPoints2),
                                  maxRansacIterations,
                                  std::ref(ransacSets),
                                  sigma,
                                  std::ref(scoreFundamental),
                                  std::ref(vbMatchesInliersF),
                                  std::ref(fundamental));

    // Wait until both threads have finished
    threadHomography.join();
    threadFundamental.join();

    // Compute ratio of scores
    r32 ratioHomographyToFundamental = scoreHomography / (scoreHomography + scoreFundamental);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if (ratioHomographyToFundamental > 0.40)
    {
        result = reconstructHomography(matches,
                                       undistortedKeyPoints1,
                                       undistortedKeyPoints2,
                                       descriptors1,
                                       descriptors2,
                                       sigma,
                                       matched,
                                       homography,
                                       cameraMat,
                                       rcw,
                                       tcw,
                                       initialPoints,
                                       keyPointTriangulatedFlags,
                                       1.0,
                                       50);
        printf("Assuming homography (%f)\n", ratioHomographyToFundamental);
    }
    else
    {
        result = reconstructFundamental(matches,
                                        undistortedKeyPoints1,
                                        undistortedKeyPoints2,
                                        descriptors1,
                                        descriptors2,
                                        sigma,
                                        vbMatchesInliersF,
                                        fundamental,
                                        cameraMat,
                                        rcw,
                                        tcw,
                                        initialPoints,
                                        keyPointTriangulatedFlags,
                                        1.0,
                                        50);
        printf("Assuming fundamental (%f)\n", ratioHomographyToFundamental);
    }

    return result;
}

static void setMapPointToBad(MapPoint* mapPoint)
{
    mapPoint->bad = true;

    std::map<KeyFrame*, i32> observations = mapPoint->observations;
    mapPoint->observations.clear();

    for (std::map<KeyFrame*, i32>::iterator observationIterator = observations.begin(), observationsEnd = observations.end();
         observationIterator != observationsEnd;
         observationIterator++)
    {
        KeyFrame* keyFrame                                     = observationIterator->first;
        keyFrame->mapPointMatches[observationIterator->second] = nullptr;
    }
}

static void addMapPointObservation(MapPoint* mapPoint,
                                   KeyFrame* observingKeyFrame,
                                   i32       keyPointIndex)
{
    if (mapPoint->observations.count(observingKeyFrame))
    {
        return;
    }

    mapPoint->observations[observingKeyFrame] = keyPointIndex;
}

static void eraseMapPointObservation(MapPoint* mapPoint,
                                     KeyFrame* observingKeyFrame)
{
    bool32 bad = false;
    {
        //unique_lock<mutex> lock(mMutexFeatures);
        if (mapPoint->observations.count(observingKeyFrame))
        {
            int idx = mapPoint->observations[observingKeyFrame];

            mapPoint->observations.erase(observingKeyFrame);

            if (mapPoint->referenceKeyFrame == observingKeyFrame)
            {
                mapPoint->referenceKeyFrame = mapPoint->observations.begin()->first;
            }

            // If only 2 observations or less, discard point
            if (mapPoint->observations.size() <= 2)
            {
                bad = true;
            }
        }
    }

    if (bad)
    {
        //SetBadFlag();
        setMapPointToBad(mapPoint);
    }
}

static void eraseKeyFrameMapPointMatch(KeyFrame* keyFrame,
                                       MapPoint* mapPoint)
{
    if (mapPoint->observations.count(keyFrame))
    {
        i32 keyPointIndex                        = mapPoint->observations[keyFrame];
        keyFrame->mapPointMatches[keyPointIndex] = static_cast<MapPoint*>(NULL);
    }
}

static void runLocalMapping(LocalMappingState*                 localMapping,
                            i32                                keyFrameCount,
                            std::set<MapPoint*>&               mapPoints,
                            std::set<KeyFrame*>&               keyFrames,
                            OrbExtractionParameters&           orbExtractionParameters,
                            r32                                fx,
                            r32                                fy,
                            r32                                cx,
                            r32                                cy,
                            r32                                invfx,
                            r32                                invfy,
                            cv::Mat&                           cameraMat,
                            i32&                               nextMapPointIndex,
                            ORBVocabulary*                     orbVocabulary,
                            std::vector<std::list<KeyFrame*>>& invertedKeyFrameFile)
{
    if (!localMapping->newKeyFrames.empty())
    {
        KeyFrame* keyFrame = localMapping->newKeyFrames.front();
        localMapping->newKeyFrames.pop_front();

        computeBoW(orbVocabulary, keyFrame);

        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (mapPoint->bad) continue;

                mapPoint->observations[keyFrame] = i;

                calculateMapPointNormalAndDepth(mapPoint->position,
                                                mapPoint->observations,
                                                keyFrame,
                                                orbExtractionParameters.scaleFactors,
                                                orbExtractionParameters.numberOfScaleLevels,
                                                &mapPoint->minDistance,
                                                &mapPoint->maxDistance,
                                                &mapPoint->normalVector);
                computeBestDescriptorFromObservations(mapPoint->observations,
                                                      &mapPoint->descriptor);
            }
        }

        updateKeyFrameConnections(keyFrame);
        keyFrames.insert(keyFrame);

        // TODO(jan): this does not belong here! move it to loopclosing
        addKeyFrameToInvertedFile(keyFrame, invertedKeyFrameFile);

        { // MapPointCulling
            std::list<MapPoint*>::iterator newMapPointIterator = localMapping->newMapPoints.begin();

            const i32 observationThreshold = 2;

            while (newMapPointIterator != localMapping->newMapPoints.end())
            {
                MapPoint* mapPoint = *newMapPointIterator;

                if (mapPoint->bad)
                {
                    newMapPointIterator = localMapping->newMapPoints.erase(newMapPointIterator);
                }
                else
                {
                    r32 foundRatio = (r32)(mapPoint->foundInKeyFrameCounter) / (r32)(mapPoint->visibleInKeyFrameCounter);
                    if (foundRatio < 0.25f)
                    {
                        setMapPointToBad(mapPoint);
                        newMapPointIterator = localMapping->newMapPoints.erase(newMapPointIterator);
                    }
                    else if ((keyFrameCount - mapPoint->referenceKeyFrame->index) >= 2 && mapPoint->observations.size() <= observationThreshold)
                    {
                        setMapPointToBad(mapPoint);
                        newMapPointIterator = localMapping->newMapPoints.erase(newMapPointIterator);
                    }
                    else if ((keyFrameCount - mapPoint->referenceKeyFrame->index) >= 3)
                    {
                        newMapPointIterator = localMapping->newMapPoints.erase(newMapPointIterator);
                    }
                    else
                    {
                        newMapPointIterator++;
                    }
                }
            }
        }

        i32 newMapPointCount = createNewMapPoints(keyFrame,
                                                  keyFrame->orderedConnectedKeyFrames,
                                                  fx,
                                                  fy,
                                                  cx,
                                                  cy,
                                                  invfx,
                                                  invfy,
                                                  orbExtractionParameters.numberOfScaleLevels,
                                                  orbExtractionParameters.scaleFactor,
                                                  cameraMat,
                                                  orbExtractionParameters.sigmaSquared,
                                                  orbExtractionParameters.scaleFactors,
                                                  mapPoints,
                                                  nextMapPointIndex,
                                                  localMapping->newMapPoints);

        printf("Created %i new mapPoints\n", newMapPointCount);

        bool abortBA = false;

        // TODO(jan): check if stop was requested
        if (localMapping->newKeyFrames.empty())
        {
            if (keyFrames.size() > 2)
            {
                { // Local Bundle Adjustment
                    bool* stopFlag = &abortBA;

                    // Local KeyFrames: First Breath Search from Current Keyframe
                    std::list<KeyFrame*> localKeyFrames;

                    localKeyFrames.push_back(keyFrame);
                    keyFrame->localBundleAdjustmentKeyFrameIndex = keyFrame->index;

                    const std::vector<KeyFrame*> neighborKeyFrames = keyFrame->orderedConnectedKeyFrames;
                    for (int i = 0, iend = neighborKeyFrames.size(); i < iend; i++)
                    {
                        KeyFrame* neighborKeyFrame                           = neighborKeyFrames[i];
                        neighborKeyFrame->localBundleAdjustmentKeyFrameIndex = keyFrame->index;
                        if (!neighborKeyFrame->bad)
                        {
                            localKeyFrames.push_back(neighborKeyFrame);
                        }
                    }

                    // Local MapPoints seen in Local KeyFrames
                    std::list<MapPoint*> localMapPoints;
                    for (list<KeyFrame*>::iterator keyFrameIterator = localKeyFrames.begin(), keyFrameIteratorEnd = localKeyFrames.end();
                         keyFrameIterator != keyFrameIteratorEnd;
                         keyFrameIterator++)
                    {
                        std::vector<MapPoint*> mapPoints = (*keyFrameIterator)->mapPointMatches;
                        for (std::vector<MapPoint*>::iterator mapPointIterator = mapPoints.begin(), mapPointIteratorEnd = mapPoints.end();
                             mapPointIterator != mapPointIteratorEnd;
                             mapPointIterator++)
                        {
                            MapPoint* mapPoint = *mapPointIterator;
                            if (mapPoint)
                            {
                                if (!mapPoint->bad)
                                    if (mapPoint->localBundleAdjustmentKeyFrameIndex != keyFrame->index)
                                    {
                                        localMapPoints.push_back(mapPoint);
                                        mapPoint->localBundleAdjustmentKeyFrameIndex = keyFrame->index;
                                    }
                            }
                        }
                    }

                    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
                    std::list<KeyFrame*> fixedKeyFrames;
                    for (std::list<MapPoint*>::iterator mapPointIterator = localMapPoints.begin(), mapPointIteratorEnd = localMapPoints.end();
                         mapPointIterator != mapPointIteratorEnd;
                         mapPointIterator++)
                    {
                        std::map<KeyFrame*, i32> observations = (*mapPointIterator)->observations;
                        for (std::map<KeyFrame*, i32>::iterator observationIterator = observations.begin(), observationIteratorEnd = observations.end();
                             observationIterator != observationIteratorEnd;
                             observationIterator++)
                        {
                            KeyFrame* fixedKeyFrame = observationIterator->first;

                            if (fixedKeyFrame->localBundleAdjustmentKeyFrameIndex != keyFrame->index && fixedKeyFrame->localBundleAdjustmentFixedKeyFrameIndex != keyFrame->index)
                            {
                                fixedKeyFrame->localBundleAdjustmentFixedKeyFrameIndex = keyFrame->index;
                                if (!fixedKeyFrame->bad)
                                {
                                    fixedKeyFrames.push_back(fixedKeyFrame);
                                }
                            }
                        }
                    }

                    // Setup optimizer
                    g2o::SparseOptimizer                    optimizer;
                    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

                    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

                    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

                    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
                    optimizer.setAlgorithm(solver);

                    if (stopFlag)
                        optimizer.setForceStopFlag(stopFlag);

                    unsigned long maxKFid = 0;

                    // Set Local KeyFrame vertices
                    for (std::list<KeyFrame*>::iterator keyFrameIterator = localKeyFrames.begin(), keyFrameIteratorEnd = localKeyFrames.end();
                         keyFrameIterator != keyFrameIteratorEnd;
                         keyFrameIterator++)
                    {
                        KeyFrame*             localKeyFrame = *keyFrameIterator;
                        g2o::VertexSE3Expmap* vertex        = new g2o::VertexSE3Expmap();
                        g2o::SE3Quat          quat          = convertCvMatToG2OSE3Quat(localKeyFrame->cTw);
                        vertex->setEstimate(quat);
                        vertex->setId(localKeyFrame->index);
                        vertex->setFixed(localKeyFrame->index == 0);
                        optimizer.addVertex(vertex);
                        if (localKeyFrame->index > maxKFid)
                        {
                            maxKFid = localKeyFrame->index;
                        }
                    }

                    // Set Fixed KeyFrame vertices
                    for (list<KeyFrame*>::iterator keyFrameIterator = fixedKeyFrames.begin(), lend = fixedKeyFrames.end();
                         keyFrameIterator != lend;
                         keyFrameIterator++)
                    {
                        KeyFrame*             fixedKeyFrame = *keyFrameIterator;
                        g2o::VertexSE3Expmap* vertex        = new g2o::VertexSE3Expmap();
                        g2o::SE3Quat          quat          = convertCvMatToG2OSE3Quat(fixedKeyFrame->cTw);
                        vertex->setEstimate(quat);
                        vertex->setId(fixedKeyFrame->index);
                        vertex->setFixed(true);
                        optimizer.addVertex(vertex);
                        if (fixedKeyFrame->index > maxKFid)
                        {
                            maxKFid = fixedKeyFrame->index;
                        }
                    }

                    // Set MapPoint vertices
                    const int nExpectedSize = (localKeyFrames.size() + fixedKeyFrames.size()) * localMapPoints.size();

                    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
                    vpEdgesMono.reserve(nExpectedSize);

                    vector<KeyFrame*> vpEdgeKFMono;
                    vpEdgeKFMono.reserve(nExpectedSize);

                    vector<MapPoint*> vpMapPointEdgeMono;
                    vpMapPointEdgeMono.reserve(nExpectedSize);

                    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
                    vpEdgesStereo.reserve(nExpectedSize);

                    vector<KeyFrame*> vpEdgeKFStereo;
                    vpEdgeKFStereo.reserve(nExpectedSize);

                    vector<MapPoint*> vpMapPointEdgeStereo;
                    vpMapPointEdgeStereo.reserve(nExpectedSize);

                    const float thHuberMono   = sqrt(5.991);
                    const float thHuberStereo = sqrt(7.815);

                    for (list<MapPoint*>::iterator mapPointIterator = localMapPoints.begin(), mapPointIteratorEnd = localMapPoints.end();
                         mapPointIterator != mapPointIteratorEnd;
                         mapPointIterator++)
                    {
                        MapPoint*               localMapPoint = *mapPointIterator;
                        g2o::VertexSBAPointXYZ* vertex        = new g2o::VertexSBAPointXYZ();
                        Eigen::Vector3d         vector3d      = convertCvMatToEigenVector3D(localMapPoint->position);
                        vertex->setEstimate(vector3d);
                        int id = localMapPoint->index + maxKFid + 1;
                        vertex->setId(id);
                        vertex->setMarginalized(true);
                        optimizer.addVertex(vertex);

                        const map<KeyFrame*, i32> observations = localMapPoint->observations;

                        //Set edges
                        for (map<KeyFrame*, i32>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;

                            if (!pKFi->bad)
                            {
                                const cv::KeyPoint& kpUn = pKFi->undistortedKeyPoints[mit->second];

                                Eigen::Matrix<double, 2, 1> obs;
                                obs << kpUn.pt.x, kpUn.pt.y;

                                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->index)));
                                e->setMeasurement(obs);
                                const float& invSigma2 = orbExtractionParameters.inverseSigmaSquared[kpUn.octave];
                                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(thHuberMono);

                                e->fx = fx;
                                e->fy = fy;
                                e->cx = cx;
                                e->cy = cy;

                                optimizer.addEdge(e);
                                vpEdgesMono.push_back(e);
                                vpEdgeKFMono.push_back(pKFi);
                                vpMapPointEdgeMono.push_back(localMapPoint);
                            }
                        }
                    }

                    if (stopFlag)
                    {
                        if (*stopFlag)
                        {
                            return;
                        }
                    }

                    optimizer.initializeOptimization();
                    optimizer.optimize(5);

                    bool doMore = true;

                    if (stopFlag)
                    {
                        if (*stopFlag)
                        {
                            doMore = false;
                        }
                    }

                    if (doMore)
                    {

                        // Check inlier observations
                        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
                        {
                            g2o::EdgeSE3ProjectXYZ* e   = vpEdgesMono[i];
                            MapPoint*               pMP = vpMapPointEdgeMono[i];

                            if (pMP->bad)
                                continue;

                            if (e->chi2() > 5.991 || !e->isDepthPositive())
                            {
                                e->setLevel(1);
                            }

                            e->setRobustKernel(0);
                        }

                        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
                        {
                            g2o::EdgeStereoSE3ProjectXYZ* e   = vpEdgesStereo[i];
                            MapPoint*                     pMP = vpMapPointEdgeStereo[i];

                            if (pMP->bad)
                                continue;

                            if (e->chi2() > 7.815 || !e->isDepthPositive())
                            {
                                e->setLevel(1);
                            }

                            e->setRobustKernel(0);
                        }

                        // Optimize again without the outliers

                        optimizer.initializeOptimization(0);
                        optimizer.optimize(10);
                    }

                    vector<pair<KeyFrame*, MapPoint*>> vToErase;
                    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

                    // Check inlier observations
                    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
                    {
                        g2o::EdgeSE3ProjectXYZ* e   = vpEdgesMono[i];
                        MapPoint*               pMP = vpMapPointEdgeMono[i];

                        if (pMP->bad)
                            continue;

                        if (e->chi2() > 5.991 || !e->isDepthPositive())
                        {
                            KeyFrame* pKFi = vpEdgeKFMono[i];
                            vToErase.push_back(make_pair(pKFi, pMP));
                        }
                    }

                    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
                    {
                        g2o::EdgeStereoSE3ProjectXYZ* e   = vpEdgesStereo[i];
                        MapPoint*                     pMP = vpMapPointEdgeStereo[i];

                        if (pMP->bad)
                            continue;

                        if (e->chi2() > 7.815 || !e->isDepthPositive())
                        {
                            KeyFrame* pKFi = vpEdgeKFStereo[i];
                            vToErase.push_back(make_pair(pKFi, pMP));
                        }
                    }

                    // Get Map Mutex
                    //unique_lock<mutex> lock(pMap->mMutexMapUpdate);

                    if (!vToErase.empty())
                    {
                        for (size_t i = 0; i < vToErase.size(); i++)
                        {
                            KeyFrame* pKFi = vToErase[i].first;
                            MapPoint* pMPi = vToErase[i].second;

                            eraseKeyFrameMapPointMatch(pKFi, pMPi);
                            eraseMapPointObservation(pMPi, pKFi);
                        }
                    }

                    // Recover optimized data

                    //Keyframes
                    for (list<KeyFrame*>::iterator lit = localKeyFrames.begin(), lend = localKeyFrames.end(); lit != lend; lit++)
                    {
                        KeyFrame*             pKF     = *lit;
                        g2o::VertexSE3Expmap* vSE3    = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->index));
                        g2o::SE3Quat          SE3quat = vSE3->estimate();
                        updatePoseMatrices(convertG2OSE3QuatToCvMat(SE3quat), pKF->cTw, pKF->wTc, pKF->worldOrigin);
                    }

                    //Points
                    for (list<MapPoint*>::iterator lit = localMapPoints.begin(), lend = localMapPoints.end(); lit != lend; lit++)
                    {
                        MapPoint*               pMP              = *lit;
                        g2o::VertexSBAPointXYZ* vPoint           = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->index + maxKFid + 1));
                        cv::Mat                 mapPointPosition = convertEigenVector3DToCvMat(vPoint->estimate());
                        mapPointPosition.copyTo(pMP->position);
                        calculateMapPointNormalAndDepth(pMP->position,
                                                        pMP->observations,
                                                        pMP->referenceKeyFrame,
                                                        orbExtractionParameters.scaleFactors,
                                                        orbExtractionParameters.numberOfScaleLevels,
                                                        &pMP->minDistance,
                                                        &pMP->maxDistance,
                                                        &pMP->normalVector);
                    }
                }
            }

            { // Keyframe culling
                // std::cout << "[LocalMapping] KeyFrameCulling" << std::endl;
                // Check redundant keyframes (only local keyframes)
                // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
                // in at least other 3 keyframes (in the same or finer scale)
                // We only consider close stereo points
                std::vector<KeyFrame*> localKeyFrames = keyFrame->orderedConnectedKeyFrames;

                for (std::vector<KeyFrame*>::iterator vit = localKeyFrames.begin(), vend = localKeyFrames.end();
                     vit != vend;
                     vit++)
                {
                    KeyFrame* localKeyFrame = *vit;
                    if (localKeyFrame->index == 0) continue;

                    const std::vector<MapPoint*> mapPoints = localKeyFrame->mapPointMatches;

                    i32       nObs                   = 3;
                    const i32 thObs                  = nObs;
                    i32       nRedundantObservations = 0;
                    i32       nMPs                   = 0;
                    for (size_t i = 0, iend = mapPoints.size(); i < iend; i++)
                    {
                        MapPoint* mapPoint = mapPoints[i];
                        if (mapPoint)
                        {
                            if (!mapPoint->bad)
                            {
                                nMPs++;
                                if (mapPoint->observations.size() > thObs)
                                {
                                    const int&               scaleLevel   = localKeyFrame->undistortedKeyPoints[i].octave;
                                    std::map<KeyFrame*, i32> observations = mapPoint->observations;
                                    int                      nObs         = 0;
                                    for (std::map<KeyFrame*, i32>::const_iterator mit = observations.begin(), mend = observations.end();
                                         mit != mend;
                                         mit++)
                                    {
                                        KeyFrame* pKFi = mit->first;
                                        if (pKFi == localKeyFrame)
                                            continue;
                                        const int& scaleLeveli = pKFi->undistortedKeyPoints[mit->second].octave;

                                        if (scaleLeveli <= scaleLevel + 1)
                                        {
                                            nObs++;
                                            if (nObs >= thObs)
                                            {
                                                break;
                                            }
                                        }
                                    }
                                    if (nObs >= thObs)
                                    {
                                        nRedundantObservations++;
                                    }
                                }
                            }
                        }
                    }

                    if (nRedundantObservations > 0.9 * nMPs)
                    {
                        localKeyFrame->bad = true;
                    }
                }
            }
        }
    }
}

static std::vector<KeyFrame*> detectRelocalizationCandidates(Frame*                             currentFrame,
                                                             std::vector<std::list<KeyFrame*>>& invertedKeyFrameFile,
                                                             const ORBVocabulary*               orbVocabulary)
{
    std::vector<KeyFrame*> result;

    std::list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        //unique_lock<mutex> lock(mMutex);
        // TODO(jan): mutex

        for (DBoW2::BowVector::const_iterator vit = currentFrame->bowVector.begin(), vend = currentFrame->bowVector.end();
             vit != vend;
             vit++)
        {
            std::list<KeyFrame*>& lKFs = invertedKeyFrameFile[vit->first];

            for (std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
                 lit != lend;
                 lit++)
            {
                KeyFrame* pKFi = *lit;
                if (pKFi->relocalizationData.queryId != currentFrame->id)
                {
                    pKFi->relocalizationData.words   = 0;
                    pKFi->relocalizationData.queryId = currentFrame->id;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->relocalizationData.words++;
            }
        }
    }

    if (lKFsSharingWords.empty()) return result;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
         lit != lend;
         lit++)
    {
        if ((*lit)->relocalizationData.words > maxCommonWords)
            maxCommonWords = (*lit)->relocalizationData.words;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, KeyFrame*>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if (pKFi->relocalizationData.words > minCommonWords)
        {
            nscores++;
            r32 si                         = orbVocabulary->score(currentFrame->bowVector, pKFi->bowVector);
            pKFi->relocalizationData.score = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty()) return result;

    list<pair<float, KeyFrame*>> lAccScoreAndMatch;
    float                        bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, KeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        KeyFrame*         pKFi     = it->second;
        vector<KeyFrame*> vpNeighs = getBestCovisibilityKeyFrames(10, pKFi->orderedConnectedKeyFrames);

        float     bestScore = it->first;
        float     accScore  = bestScore;
        KeyFrame* pBestKF   = pKFi;
        for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if (pKF2->relocalizationData.queryId != currentFrame->id) continue;

            accScore += pKF2->relocalizationData.score;
            if (pKF2->relocalizationData.score > bestScore)
            {
                pBestKF   = pKF2;
                bestScore = pKF2->relocalizationData.score;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
        {
            bestAccScore = accScore;
        }
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float          minScoreToRetain = 0.75f * bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    result.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, KeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float& si = it->first;
        if (si > minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                result.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return result;
}

static bool32 relocalize(Frame*                             currentFrame,
                         const ORBVocabulary*               orbVocabulary,
                         std::vector<std::list<KeyFrame*>>& invertedKeyFrameFile,
                         const std::vector<r32>&            sigmaSquared,
                         const std::vector<r32>             inverseSigmaSquared,
                         const GridConstraints              gridConstraints,
                         const std::vector<r32>&            scaleFactors,
                         const i32                          numberOfScaleLevels,
                         const r32                          logScaleFactor,
                         const r32                          fx,
                         const r32                          fy,
                         const r32                          cx,
                         const r32                          cy,
                         i32*                               lastRelocalizationId)
{
    // Compute Bag of Words Vector
    computeBoW(orbVocabulary, currentFrame);

    // Relocalization is performed when tracking is lost
    std::vector<KeyFrame*> vpCandidateKFs = detectRelocalizationCandidates(currentFrame,
                                                                           invertedKeyFrameFile,
                                                                           orbVocabulary);

    printf("relocalizing: %i candidates\n", vpCandidateKFs.size());

    //std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty()) return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver

    //std::vector<PnPsolver*> vpPnPsolvers;
    //vpPnPsolvers.resize(nKFs);

    std::vector<PnPSolver> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    std::vector<std::vector<MapPoint*>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    std::vector<bool32> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->bad)
        {
            vbDiscarded[i] = true;
        }
        else
        {
            int nmatches = findMapPointMatchesByBoW(pKF->featureVector,
                                                    currentFrame->featureVector,
                                                    pKF->mapPointMatches,
                                                    pKF->undistortedKeyPoints,
                                                    currentFrame->keyPoints,
                                                    pKF->descriptors,
                                                    currentFrame->descriptors,
                                                    true,
                                                    vvpMapPointMatches[i],
                                                    0.75f);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPSolver pnpSolver = initializePnPSolver(currentFrame->mapPointMatches,
                                                          currentFrame->undistortedKeyPoints,
                                                          sigmaSquared,
                                                          vvpMapPointMatches[i],
                                                          0.99,
                                                          10,
                                                          300,
                                                          4,
                                                          0.5,
                                                          5.991,
                                                          fx,
                                                          fy,
                                                          cx,
                                                          cy);
                vpPnPsolvers[i]     = pnpSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool32 bMatch = false;
    //ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i]) continue;

            // Perform 5 Ransac Iterations
            std::vector<bool32> vbInliers;
            int                 nInliers;
            bool32              bNoMore;

            PnPSolver solver = vpPnPsolvers[i];
            cv::Mat   Tcw    = solvePnP(&solver, 5, &bNoMore, vbInliers, &nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(currentFrame->cTw);

                std::set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        currentFrame->mapPointMatches[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        currentFrame->mapPointMatches[j] = NULL;
                }

                i32 nGood = optimizePose(inverseSigmaSquared, fx, fy, cx, cy, currentFrame);

                if (nGood < 10) continue;

                for (int io = 0; io < currentFrame->numberOfKeyPoints; io++)
                {
                    if (currentFrame->mapPointIsOutlier[io])
                    {
                        currentFrame->mapPointMatches[io] = NULL;
                    }
                }

                // If few inliers, search by projection in a coarse window and optimize again:
                //ghm1: mappoints seen in the keyframe which was found as candidate via BoW-search are projected into
                //the current frame using the position that was calculated using the matches from BoW matcher
                if (nGood < 50)
                {
                    KeyFrame* candidateKeyFrame = vpCandidateKFs[i];

                    i32 nadditional = searchMapPointsByProjectionOfCandidateKeyFrameMapPoints(currentFrame->cTw,
                                                                                              currentFrame->keyPointIndexGrid,
                                                                                              currentFrame->numberOfKeyPoints,
                                                                                              currentFrame->undistortedKeyPoints,
                                                                                              currentFrame->descriptors,
                                                                                              currentFrame->mapPointMatches,
                                                                                              candidateKeyFrame->mapPointMatches,
                                                                                              candidateKeyFrame->undistortedKeyPoints,
                                                                                              sFound,
                                                                                              gridConstraints,
                                                                                              scaleFactors,
                                                                                              numberOfScaleLevels,
                                                                                              logScaleFactor,
                                                                                              fx,
                                                                                              fy,
                                                                                              cx,
                                                                                              cy,
                                                                                              10,
                                                                                              100,
                                                                                              true);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = optimizePose(inverseSigmaSquared,
                                             fx,
                                             fy,
                                             cx,
                                             cy,
                                             currentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < currentFrame->numberOfKeyPoints; ip++)
                            {
                                if (currentFrame->mapPointMatches[ip])
                                {
                                    sFound.insert(currentFrame->mapPointMatches[ip]);
                                }
                            }

                            nadditional = searchMapPointsByProjectionOfCandidateKeyFrameMapPoints(currentFrame->cTw,
                                                                                                  currentFrame->keyPointIndexGrid,
                                                                                                  currentFrame->numberOfKeyPoints,
                                                                                                  currentFrame->undistortedKeyPoints,
                                                                                                  currentFrame->descriptors,
                                                                                                  currentFrame->mapPointMatches,
                                                                                                  candidateKeyFrame->mapPointMatches,
                                                                                                  candidateKeyFrame->undistortedKeyPoints,
                                                                                                  sFound,
                                                                                                  gridConstraints,
                                                                                                  scaleFactors,
                                                                                                  numberOfScaleLevels,
                                                                                                  logScaleFactor,
                                                                                                  fx,
                                                                                                  fy,
                                                                                                  cx,
                                                                                                  cy,
                                                                                                  3,
                                                                                                  64,
                                                                                                  true);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = optimizePose(inverseSigmaSquared,
                                                     fx,
                                                     fy,
                                                     cx,
                                                     cy,
                                                     currentFrame);

                                for (int io = 0; io < currentFrame->numberOfKeyPoints; io++)
                                {
                                    if (currentFrame->mapPointIsOutlier[io])
                                    {
                                        currentFrame->mapPointMatches[io] = NULL;
                                    }
                                }
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        *lastRelocalizationId = currentFrame->id;
        return true;
    }
}

void WAI::ModeOrbSlam2DataOriented::notifyUpdate()
{
    _state.frameCounter++;

    // TODO(jan): check reset

    switch (_state.status)
    {
        case OrbSlamStatus_Initializing:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            if (!_state.initialFrameSet)
            {
                computeGridConstraints(cameraFrame,
                                       cameraMat,
                                       distortionMat,
                                       &_state.gridConstraints);

                _state.fx    = cameraMat.at<r32>(0, 0);
                _state.fy    = cameraMat.at<r32>(1, 1);
                _state.cx    = cameraMat.at<r32>(0, 2);
                _state.cy    = cameraMat.at<r32>(1, 2);
                _state.invfx = 1.0f / _state.fx;
                _state.invfy = 1.0f / _state.fy;
            }

            Frame currentFrame;
            initializeFrame(&currentFrame,
                            _state.nextFrameId,
                            cameraFrame,
                            cameraMat,
                            distortionMat,
                            _state.initializationOrbExtractionParameters,
                            _state.gridConstraints);

#if 0
            if (_state.frameCounter == 20)
            {
                cv::Mat showFrame = cameraFrame.clone();

                for (i32 i = 0; i < currentFrame.undistortedKeyPoints.size(); i++)
                {
                    cv::KeyPoint kP = currentFrame.undistortedKeyPoints[i];
                    cv::rectangle(showFrame,
                                  cv::Rect((int)kP.pt.x - 3, (int)kP.pt.y - 3, 7, 7),
                                  cv::Scalar(0, 0, 255));
                }

                cv::imshow("Frame 20", showFrame);
                cv::waitKey(0);
            }
#endif

            if (!_state.initialFrameSet)
            {
                if (currentFrame.keyPoints.size() > 100)
                {
                    initializeFrame(&_state.initialFrame, currentFrame);
                    initializeFrame(&_state.lastFrame, currentFrame);

                    _state.previouslyMatchedKeyPoints.resize(currentFrame.numberOfKeyPoints);
                    for (i32 i = 0; i < currentFrame.numberOfKeyPoints; i++)
                    {
                        _state.previouslyMatchedKeyPoints[i] = currentFrame.undistortedKeyPoints[i].pt;
                    }

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);

                    _state.initialFrameSet = true;

                    printf("First initialization keyFrame at frame %i\n", _state.frameCounter);
                }
            }
            else
            {
                if (currentFrame.keyPoints.size() > 100)
                {
                    // NOTE(jan): initialization matches contains the index of the matched keypoint
                    // of the current keyframe for every keypoint of the reference keyframe or
                    // -1 if not matched.
                    // currentKeyFrame->keyPoints[initializationMatches[i]] is matched to referenceKeyFrame->keyPoints[i]
                    bool32 checkOrientation                      = true;
                    r32    shortestToSecondShortestDistanceRatio = 0.9f;

                    i32 numberOfMatches = 0;

                    numberOfMatches = findInitializationMatches(_state.initialFrame.undistortedKeyPoints,
                                                                currentFrame.undistortedKeyPoints,
                                                                _state.initialFrame.descriptors,
                                                                currentFrame.descriptors,
                                                                currentFrame.keyPointIndexGrid,
                                                                _state.previouslyMatchedKeyPoints,
                                                                _state.gridConstraints,
                                                                shortestToSecondShortestDistanceRatio,
                                                                checkOrientation,
                                                                100,
                                                                _state.initializationMatches);

                    // update prev matched
                    for (size_t i1 = 0, iend1 = _state.initializationMatches.size();
                         i1 < iend1;
                         i1++)
                    {
                        if (_state.initializationMatches[i1] >= 0)
                        {
                            _state.previouslyMatchedKeyPoints[i1] = currentFrame.undistortedKeyPoints[_state.initializationMatches[i1]].pt;
                        }
                    }

                    //ghm1: decorate image with tracked matches
                    for (u32 i = 0; i < _state.initializationMatches.size(); i++)
                    {
                        if (_state.initializationMatches[i] >= 0)
                        {
                            cv::line(_camera->getImageRGB(),
                                     _state.initialFrame.keyPoints[i].pt,
                                     currentFrame.keyPoints[_state.initializationMatches[i]].pt,
                                     cv::Scalar(0, 255, 0));
                        }
                    }

                    // Check if there are enough matches
                    if (numberOfMatches >= 100)
                    {
                        cv::Mat                  crw;                       // Current Camera Rotation
                        cv::Mat                  ctw;                       // Current Camera Translation
                        std::vector<bool32>      keyPointTriangulatedFlags; // Triangulated Correspondences (mvIniMatches)
                        std::vector<cv::Point3f> initialPoints;

                        bool32 validModelFound = computeGeometricalModel(_state.initialFrame.undistortedKeyPoints,
                                                                         currentFrame.undistortedKeyPoints,
                                                                         _state.initialFrame.descriptors,
                                                                         currentFrame.descriptors,
                                                                         _state.initializationMatches,
                                                                         cameraMat,
                                                                         crw,
                                                                         ctw,
                                                                         keyPointTriangulatedFlags,
                                                                         initialPoints);

                        if (validModelFound)
                        {
                            for (i32 i = 0; i < _state.initializationMatches.size(); i++)
                            {
                                if (_state.initializationMatches[i] >= 0 && !keyPointTriangulatedFlags[i])
                                {
                                    _state.initializationMatches[i] = -1;
                                    numberOfMatches--;
                                }
                            }

                            updatePoseMatrices(cv::Mat::eye(4, 4, CV_32F),
                                               _state.initialFrame.cTw,
                                               _state.initialFrame.wTc,
                                               _state.initialFrame.worldOrigin);

                            cv::Mat cTw = cv::Mat::eye(4, 4, CV_32F);
                            crw.copyTo(cTw.rowRange(0, 3).colRange(0, 3));
                            ctw.copyTo(cTw.rowRange(0, 3).col(3));

                            updatePoseMatrices(cTw,
                                               currentFrame.cTw,
                                               currentFrame.wTc,
                                               currentFrame.worldOrigin);

                            // CreateInitialMapMonocular
                            KeyFrame* initialKeyFrame = new KeyFrame();
                            KeyFrame* currentKeyFrame = new KeyFrame();

                            initializeKeyFrame(initialKeyFrame, _state.initialFrame, _state.nextKeyFrameId);
                            initializeKeyFrame(currentKeyFrame, currentFrame, _state.nextKeyFrameId);

                            computeBoW(_state.orbVocabulary, initialKeyFrame);
                            computeBoW(_state.orbVocabulary, currentKeyFrame);

                            _state.keyFrames.insert(initialKeyFrame);
                            _state.keyFrames.insert(currentKeyFrame);

                            for (size_t i = 0, iend = _state.initializationMatches.size(); i < iend; i++)
                            {
                                if (_state.initializationMatches[i] < 0) continue;

                                cv::Mat worldPosition(initialPoints[i]);

                                MapPoint* mapPoint = new MapPoint();

                                initializeMapPoint(mapPoint, currentKeyFrame, worldPosition, _state.nextMapPointId);

                                initialKeyFrame->mapPointMatches[i]                               = mapPoint;
                                currentKeyFrame->mapPointMatches[_state.initializationMatches[i]] = mapPoint;

                                addMapPointObservation(mapPoint, initialKeyFrame, i);
                                addMapPointObservation(mapPoint, currentKeyFrame, _state.initializationMatches[i]);

                                computeBestDescriptorFromObservations(mapPoint->observations,
                                                                      &mapPoint->descriptor);
                                calculateMapPointNormalAndDepth(mapPoint->position,
                                                                mapPoint->observations,
                                                                mapPoint->referenceKeyFrame,
                                                                _state.initializationOrbExtractionParameters.scaleFactors,
                                                                _state.initializationOrbExtractionParameters.numberOfScaleLevels,
                                                                &mapPoint->minDistance,
                                                                &mapPoint->maxDistance,
                                                                &mapPoint->normalVector);

                                currentFrame.mapPointMatches[_state.initializationMatches[i]]   = mapPoint;
                                currentFrame.mapPointIsOutlier[_state.initializationMatches[i]] = false;

                                _state.mapPoints.insert(mapPoint);
                            }

                            updateKeyFrameConnections(initialKeyFrame);
                            updateKeyFrameConnections(currentKeyFrame);

                            printf("New Map created with %i points\n", _state.mapPoints.size());

                            { // (Global) bundle adjustment
                                std::vector<KeyFrame*> keyFrames = std::vector<KeyFrame*>(_state.keyFrames.begin(), _state.keyFrames.end());
                                std::vector<MapPoint*> mapPoints = std::vector<MapPoint*>(_state.mapPoints.begin(), _state.mapPoints.end());

                                const i32 numberOfIterations = 20;
                                const i32 loopKeyframeCount  = 0;

                                std::vector<bool32> mapPointNotIncludedFlags;
                                mapPointNotIncludedFlags.resize(mapPoints.size());

                                g2o::SparseOptimizer                    optimizer;
                                g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

                                linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

                                g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

                                g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
                                optimizer.setAlgorithm(solver);

                                //if(pbStopFlag)
                                //optimizer.setForceStopFlag(pbStopFlag);

                                u64 maxKFid = 0;

                                // Set KeyFrame vertices
                                for (i32 i = 0; i < keyFrames.size(); i++)
                                {
                                    KeyFrame* keyFrame = keyFrames[i];

                                    if (keyFrame->bad) continue;

                                    g2o::SE3Quat quat = convertCvMatToG2OSE3Quat(keyFrame->cTw);

                                    g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
                                    vertex->setEstimate(quat);
                                    vertex->setId(keyFrame->index);
                                    vertex->setFixed(keyFrame->index == 0);
                                    optimizer.addVertex(vertex);

                                    if (keyFrame->index > maxKFid)
                                    {
                                        maxKFid = keyFrame->index;
                                    }
                                }

                                const r32 thHuber2D = sqrt(5.99);
                                const r32 thHuber3D = sqrt(7.815);

                                for (i32 i = 0; i < mapPoints.size(); i++)
                                {
                                    MapPoint* mapPoint = mapPoints[i];

                                    if (mapPoint->bad) continue;

                                    Eigen::Matrix<r64, 3, 1> vec3d = convertCvMatToEigenVector3D(mapPoint->position);

                                    g2o::VertexSBAPointXYZ* vertex = new g2o::VertexSBAPointXYZ();
                                    vertex->setEstimate(vec3d);
                                    const i32 id = mapPoint->index + maxKFid + 1;
                                    vertex->setId(id);
                                    vertex->setMarginalized(true);
                                    optimizer.addVertex(vertex);

                                    const std::map<KeyFrame*, i32> observations = mapPoint->observations;

                                    i32 edgeCount = 0;

                                    //SET EDGES
                                    for (std::map<KeyFrame*, i32>::const_iterator it = observations.begin(), itend = observations.end();
                                         it != itend;
                                         it++)
                                    {
                                        KeyFrame* keyFrame = it->first;

                                        if (keyFrame->bad || keyFrame->index > maxKFid) continue;

                                        edgeCount++;

                                        const cv::KeyPoint& undistortedKeyPoint = keyFrame->undistortedKeyPoints[it->second];

                                        Eigen::Matrix<r64, 2, 1> observationMatrix;
                                        observationMatrix << undistortedKeyPoint.pt.x, undistortedKeyPoint.pt.y;

                                        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();

                                        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                                        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyFrame->index)));
                                        edge->setMeasurement(observationMatrix);
                                        const r32& invSigmaSquared = _state.initializationOrbExtractionParameters.inverseSigmaSquared[undistortedKeyPoint.octave];
                                        edge->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquared);

                                        //if (bRobust)
                                        //{
                                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                        edge->setRobustKernel(rk);
                                        rk->setDelta(thHuber3D);
                                        //}

                                        edge->fx = _state.fx;
                                        edge->fy = _state.fy;
                                        edge->cx = _state.cx;
                                        edge->cy = _state.cy;

                                        optimizer.addEdge(edge);
                                    }

                                    if (edgeCount == 0)
                                    {
                                        optimizer.removeVertex(vertex);
                                        mapPointNotIncludedFlags[i] = true;
                                    }
                                    else
                                    {
                                        mapPointNotIncludedFlags[i] = false;
                                    }
                                }

                                // Optimize!
                                optimizer.initializeOptimization();
                                optimizer.optimize(numberOfIterations);

                                // Recover optimized data

                                // Keyframes
                                for (i32 i = 0; i < keyFrames.size(); i++)
                                {
                                    KeyFrame* keyFrame = keyFrames[i];

                                    if (keyFrame->bad) continue;

                                    g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
                                    g2o::SE3Quat          quat   = vertex->estimate();

                                    updatePoseMatrices(convertG2OSE3QuatToCvMat(quat), keyFrame->cTw, keyFrame->wTc, keyFrame->worldOrigin);

#if 0
                                    if (loopKeyframeCount == 0)
                                    {
                                        updatePoseMatrices(convertG2OSE3QuatToCvMat(quat), keyFrame->cTw, keyFrame->wTc, keyFrame->worldOrigin);
                                    }
                                    else
                                    {
                                        pKF->mTcwGBA.create(4, 4, CV_32F);
                                        Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                                        pKF->mnBAGlobalForKF = nLoopKF;
                                    }
#else
                                    updatePoseMatrices(convertG2OSE3QuatToCvMat(quat), keyFrame->cTw, keyFrame->wTc, keyFrame->worldOrigin);
#endif
                                }

                                // Points
                                for (size_t i = 0; i < mapPoints.size(); i++)
                                {
                                    if (mapPointNotIncludedFlags[i]) continue;

                                    MapPoint* mapPoint = mapPoints[i];

                                    if (mapPoint->bad) continue;

                                    g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + _state.nextKeyFrameId));
#if 0
                                    if (nLoopKF == 0)
                                    {
                                        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                                        pMP->UpdateNormalAndDepth();
                                    }
                                    else
                                    {
                                        pMP->mPosGBA.create(3, 1, CV_32F);
                                        Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                                        pMP->mnBAGlobalForKF = nLoopKF;
                                    }
#else
                                    cv::Mat worldPosition = convertEigenVector3DToCvMat(vertex->estimate());
                                    worldPosition.copyTo(mapPoint->position);
                                    calculateMapPointNormalAndDepth(mapPoint->position,
                                                                    mapPoint->observations,
                                                                    mapPoint->referenceKeyFrame,
                                                                    _state.initializationOrbExtractionParameters.scaleFactors,
                                                                    _state.initializationOrbExtractionParameters.numberOfScaleLevels,
                                                                    &mapPoint->minDistance,
                                                                    &mapPoint->maxDistance,
                                                                    &mapPoint->normalVector);
#endif
                                }
                            }

                            r32 medianDepth    = computeSceneMedianDepthForKeyFrame(initialKeyFrame);
                            r32 invMedianDepth = 1.0f / medianDepth;

                            // TODO(jan): is the check for tracked map points necessary,
                            // as we have the same check already higher up?
                            i32 trackedMapPoints;

                            { // WAIKeyFrame->TrackedMapPoints
                                std::vector<MapPoint*> mapPointMatches = currentKeyFrame->mapPointMatches;
                                for (i32 i = 0; i < mapPointMatches.size(); i++)
                                {
                                    MapPoint* mapPoint = mapPointMatches[i];

                                    if (!mapPoint) continue;
                                    if (mapPoint->bad) continue;

                                    if (mapPoint->observations.size() > 0)
                                    {
                                        trackedMapPoints++;
                                    }
                                }
                            }

                            if (medianDepth > 0.0f && trackedMapPoints >= 100)
                            {
                                cv::Mat scaledPose               = currentKeyFrame->cTw;
                                scaledPose.col(3).rowRange(0, 3) = scaledPose.col(3).rowRange(0, 3) * invMedianDepth;
                                updatePoseMatrices(scaledPose,
                                                   currentKeyFrame->cTw,
                                                   currentKeyFrame->wTc,
                                                   currentKeyFrame->worldOrigin);

                                // Scale points
                                std::vector<MapPoint*> allMapPoints = initialKeyFrame->mapPointMatches;
                                for (i32 i = 0; i < allMapPoints.size(); i++)
                                {
                                    if (allMapPoints[i])
                                    {
                                        MapPoint* mapPoint     = allMapPoints[i];
                                        cv::Mat   unscaledPose = mapPoint->position.clone();
                                        cv::Mat   scaledPose   = unscaledPose * invMedianDepth;
                                        scaledPose.copyTo(mapPoint->position);
                                    }
                                }

                                _state.localMapping.newKeyFrames.push_back(initialKeyFrame);
                                _state.localMapping.newKeyFrames.push_back(currentKeyFrame);

                                _state.localKeyFrames.push_back(initialKeyFrame);
                                _state.localKeyFrames.push_back(currentKeyFrame);
                                _state.localMapPoints          = std::vector<MapPoint*>(_state.mapPoints.begin(), _state.mapPoints.end());
                                _state.referenceKeyFrame       = currentKeyFrame;
                                currentFrame.referenceKeyFrame = currentKeyFrame;

                                initializeFrame(&_state.lastFrame, currentFrame);

                                // TODO(jan): save stuff for camera trajectory
                                // TODO(jan): set reference map points

                                runLocalMapping(&_state.localMapping,
                                                _state.keyFrames.size(),
                                                _state.mapPoints,
                                                _state.keyFrames,
                                                _state.initializationOrbExtractionParameters,
                                                _state.fx,
                                                _state.fy,
                                                _state.cx,
                                                _state.cy,
                                                _state.invfx,
                                                _state.invfy,
                                                cameraMat,
                                                _state.nextMapPointId,
                                                _state.orbVocabulary,
                                                _state.invertedKeyFrameFile);
                                runLocalMapping(&_state.localMapping,
                                                _state.keyFrames.size(),
                                                _state.mapPoints,
                                                _state.keyFrames,
                                                _state.initializationOrbExtractionParameters,
                                                _state.fx,
                                                _state.fy,
                                                _state.cx,
                                                _state.cy,
                                                _state.invfx,
                                                _state.invfy,
                                                cameraMat,
                                                _state.nextMapPointId,
                                                _state.orbVocabulary,
                                                _state.invertedKeyFrameFile);

                                _state.status        = OrbSlamStatus_Tracking;
                                _state.trackingWasOk = true;

                                printf("Second initialization keyFrame at frame %i\n", _state.frameCounter);

                                std::cout << initialKeyFrame->cTw << std::endl;
                                std::cout << currentKeyFrame->cTw << std::endl;
                            }
                            else
                            {
                                WAI_LOG("Wrong initialization, reseting...");

                                for (std::set<KeyFrame*>::iterator keyFrameIterator = _state.keyFrames.begin(), keyFrameIteratorEnd = _state.keyFrames.end();
                                     keyFrameIterator != keyFrameIteratorEnd;
                                     keyFrameIterator++)
                                {
                                    delete *keyFrameIterator;
                                }

                                for (std::set<MapPoint*>::iterator mapPointIterator = _state.mapPoints.begin(), mapPointIteratorEnd = _state.mapPoints.end();
                                     mapPointIterator != mapPointIteratorEnd;
                                     mapPointIterator++)
                                {
                                    delete *mapPointIterator;
                                }

                                _state.keyFrames.clear();
                                _state.mapPoints.clear();

                                _state.nextKeyFrameId = 0;
                                _state.nextFrameId    = 0;
                                _state.nextMapPointId = 0;
                            }
                        }
                    }
                    else
                    {
                        _state.initialFrameSet = false;
                    }
                }
                else
                {
                    _state.initialFrameSet = false;

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);
                }
            }
        }
        break;

        case OrbSlamStatus_Tracking:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            Frame currentFrame;
            initializeFrame(&currentFrame,
                            _state.nextFrameId,
                            cameraFrame,
                            cameraMat,
                            distortionMat,
                            _state.orbExtractionParameters,
                            _state.gridConstraints);
            _state.nextKeyFrameId++;

            KeyFrame* referenceKeyFrame = _state.referenceKeyFrame;

            bool32 trackingIsOk = false;

            if (_state.trackingWasOk)
            {
                // TODO(jan): checkReplacedInLastFrame
                // TODO(jan): velocity model

                computeBoW(_state.orbVocabulary,
                           currentFrame.descriptors,
                           currentFrame.bowVector,
                           currentFrame.featureVector);

                std::vector<MapPoint*> mapPointMatches;
                i32                    matchCount = findMapPointMatchesByBoW(referenceKeyFrame->featureVector,
                                                          currentFrame.featureVector,
                                                          referenceKeyFrame->mapPointMatches,
                                                          referenceKeyFrame->undistortedKeyPoints,
                                                          currentFrame.keyPoints,
                                                          referenceKeyFrame->descriptors,
                                                          currentFrame.descriptors,
                                                          true,
                                                          mapPointMatches,
                                                          0.7f);

                // TODO(jan): magic number
                if (matchCount > 15)
                {
                    currentFrame.mapPointMatches = mapPointMatches;
                    updatePoseMatrices(referenceKeyFrame->cTw,
                                       currentFrame.cTw,
                                       currentFrame.wTc,
                                       currentFrame.worldOrigin);

                    optimizePose(_state.orbExtractionParameters.inverseSigmaSquared,
                                 _state.fx,
                                 _state.fy,
                                 _state.cx,
                                 _state.cy,
                                 &currentFrame);

                    i32 goodMatchCount = 0;

                    // discard outliers
                    for (i32 i = 0; i < currentFrame.numberOfKeyPoints; i++)
                    {
                        MapPoint* mapPoint = currentFrame.mapPointMatches[i];
                        if (mapPoint)
                        {
                            if (currentFrame.mapPointIsOutlier[i])
                            {
                                currentFrame.mapPointMatches[i]   = nullptr;
                                currentFrame.mapPointIsOutlier[i] = false;
                                //matchCount--;
                            }
                            else if (!mapPoint->observations.empty())
                            {
                                goodMatchCount++;
                            }
                        }
                    }

                    if (goodMatchCount > 10)
                    {
                        trackingIsOk = true;
                    }

                    printf("Found %i good matches (out of %i)\n", goodMatchCount, matchCount);
                }
                else
                {
                    printf("Only found %i matches\n", matchCount);
                }
            }
            else
            {
                relocalize(&currentFrame,
                           _state.orbVocabulary,
                           _state.invertedKeyFrameFile,
                           _state.orbExtractionParameters.sigmaSquared,
                           _state.orbExtractionParameters.inverseSigmaSquared,
                           _state.gridConstraints,
                           _state.orbExtractionParameters.scaleFactors,
                           _state.orbExtractionParameters.numberOfScaleLevels,
                           _state.orbExtractionParameters.logScaleFactor,
                           _state.fx,
                           _state.fy,
                           _state.cx,
                           _state.cy,
                           &_state.lastRelocalizationFrameId);
            }

            // TODO(jan): set current frame reference keyframe

            i32 inlierMatches = 0;

            if (trackingIsOk)
            {
                {     // Track local map
                    { // update local keyframes
                        // Each map point votes for the keyframes in which it has been observed
                        std::map<KeyFrame*, i32> keyframeCounter;
                        for (i32 i = 0; i < currentFrame.numberOfKeyPoints; i++)
                        {
                            MapPoint* mapPoint = currentFrame.mapPointMatches[i];

                            if (mapPoint)
                            {
                                if (!mapPoint->bad)
                                {
                                    std::map<KeyFrame*, i32> observations = mapPoint->observations;

                                    for (std::map<KeyFrame*, i32>::const_iterator it = observations.begin(), itend = observations.end();
                                         it != itend;
                                         it++)
                                    {
                                        keyframeCounter[it->first]++;
                                    }
                                }
                                else
                                {
                                    currentFrame.mapPointMatches[i] = nullptr;
                                }
                            }
                        }

                        if (!keyframeCounter.empty())
                        {
                            i32       maxObservations             = 0;
                            KeyFrame* keyFrameWithMaxObservations = nullptr;

                            _state.localKeyFrames.clear();
                            _state.localKeyFrames.reserve(3 * keyframeCounter.size());

                            // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
                            for (std::map<KeyFrame*, i32>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
                                 it != itEnd;
                                 it++)
                            {
                                KeyFrame* keyFrame = it->first;

                                if (keyFrame->bad) continue;

                                if (it->second > maxObservations)
                                {
                                    maxObservations             = it->second;
                                    keyFrameWithMaxObservations = keyFrame;
                                }

                                _state.localKeyFrames.push_back(it->first);
                                keyFrame->trackReferenceForFrame = currentFrame.id;
                            }

                            // Include also some not-already-included keyframes that are neighbors to already-included keyframes
                            for (std::vector<KeyFrame*>::const_iterator itKF = _state.localKeyFrames.begin(), itEndKF = _state.localKeyFrames.end();
                                 itKF != itEndKF;
                                 itKF++)
                            {
                                // Limit the number of keyframes
                                if (_state.localKeyFrames.size() > 80) break;

                                KeyFrame* keyFrame = *itKF;

                                std::vector<KeyFrame*> neighborKeyFrames = getBestCovisibilityKeyFrames(10,
                                                                                                        keyFrame->orderedConnectedKeyFrames);

                                for (std::vector<KeyFrame*>::const_iterator itNeighKF = neighborKeyFrames.begin(), itEndNeighKF = neighborKeyFrames.end();
                                     itNeighKF != itEndNeighKF;
                                     itNeighKF++)
                                {
                                    KeyFrame* neighborKeyFrame = *itNeighKF;
                                    if (!neighborKeyFrame->bad)
                                    {
                                        if (neighborKeyFrame->trackReferenceForFrame != currentFrame.id)
                                        {
                                            _state.localKeyFrames.push_back(neighborKeyFrame);
                                            neighborKeyFrame->trackReferenceForFrame = currentFrame.id;
                                            break;
                                        }
                                    }
                                }

                                const std::vector<KeyFrame*> childrenKeyFrames = keyFrame->children;
                                for (std::vector<KeyFrame*>::const_iterator sit = childrenKeyFrames.begin(), send = childrenKeyFrames.end();
                                     sit != send;
                                     sit++)
                                {
                                    KeyFrame* childKeyFrame = *sit;
                                    if (!childKeyFrame->bad)
                                    {
                                        if (childKeyFrame->trackReferenceForFrame != currentFrame.id)
                                        {
                                            _state.localKeyFrames.push_back(childKeyFrame);
                                            childKeyFrame->trackReferenceForFrame = currentFrame.id;
                                            break;
                                        }
                                    }
                                }

                                KeyFrame* parentKeyFrame = keyFrame->parent;
                                if (parentKeyFrame)
                                {
                                    if (parentKeyFrame->trackReferenceForFrame != currentFrame.id)
                                    {
                                        _state.localKeyFrames.push_back(parentKeyFrame);
                                        parentKeyFrame->trackReferenceForFrame = currentFrame.id;
                                        break;
                                    }
                                }
                            }

                            if (keyFrameWithMaxObservations)
                            {
                                _state.referenceKeyFrame       = keyFrameWithMaxObservations;
                                currentFrame.referenceKeyFrame = keyFrameWithMaxObservations;
                            }
                        }
                    }

                    { // update local points
                        // TODO(jan): as we always clear the localMapPoints, is it necessary to keep it in state?
                        _state.localMapPoints.clear();

                        for (std::vector<KeyFrame*>::const_iterator itKF = _state.localKeyFrames.begin(), itEndKF = _state.localKeyFrames.end();
                             itKF != itEndKF;
                             itKF++)
                        {
                            KeyFrame*                    keyFrame        = *itKF;
                            const std::vector<MapPoint*> mapPointMatches = keyFrame->mapPointMatches;

                            for (std::vector<MapPoint*>::const_iterator itMP = mapPointMatches.begin(), itEndMP = mapPointMatches.end();
                                 itMP != itEndMP;
                                 itMP++)
                            {
                                MapPoint* mapPoint = *itMP;

                                if (!mapPoint) continue;
                                if (mapPoint->trackReferenceForFrame == currentFrame.id) continue;

                                if (!mapPoint->bad)
                                {
                                    _state.localMapPoints.push_back(mapPoint);
                                    mapPoint->trackReferenceForFrame = currentFrame.id;
                                }
                            }
                        }
                    }

                    { // search local points
                        std::vector<bool32> mapPointAlreadyMatched = std::vector<bool32>(_state.mapPoints.size(), false);

                        // Do not search map points already matched
                        for (std::vector<MapPoint*>::iterator vit = currentFrame.mapPointMatches.begin(), vend = currentFrame.mapPointMatches.end();
                             vit != vend;
                             vit++)
                        {
                            MapPoint* mapPoint = *vit;

                            if (mapPoint)
                            {
                                if (mapPoint->bad)
                                {
                                    *vit = nullptr;
                                }
                                else
                                {
                                    mapPoint->visibleInKeyFrameCounter++;
                                    mapPoint->lastFrameSeen = currentFrame.id;

                                    /*
                                    pMP->mbTrackInView   = false;
                                    */
                                }
                            }
                        }

                        i32 numberOfMapPointsToMatch = 0;

                        // Project points in frame and check its visibility
                        for (std::vector<MapPoint*>::iterator vit = _state.localMapPoints.begin(), vend = _state.localMapPoints.end();
                             vit != vend;
                             vit++)
                        {
                            MapPoint* mapPoint = *vit;

                            if (mapPoint)
                            {
                                if (mapPoint->lastFrameSeen == currentFrame.id) continue;
                                if (mapPoint->bad) continue;

                                // Project (this fills MapPoint variables for matching)
                                if (isMapPointInFrameFrustum(&currentFrame,
                                                             _state.fx,
                                                             _state.fy,
                                                             _state.cx,
                                                             _state.cy,
                                                             _state.gridConstraints.minX,
                                                             _state.gridConstraints.maxX,
                                                             _state.gridConstraints.minY,
                                                             _state.gridConstraints.maxY,
                                                             mapPoint,
                                                             0.5f,
                                                             &mapPoint->trackingInfos))
                                {
                                    mapPoint->visibleInKeyFrameCounter++;
                                    numberOfMapPointsToMatch++;
                                }
                            }
                        }

                        if (numberOfMapPointsToMatch > 0)
                        {
                            i32 threshold = 1;

                            // If the camera has been relocalised recently, perform a coarser search
                            if (currentFrame.id < _state.lastRelocalizationFrameId + 2)
                            {
                                threshold = 5;
                            }

                            i32 localMatches = searchMapPointsByProjectionOfLocalMapPoints(_state.localMapPoints,
                                                                                           _state.orbExtractionParameters.scaleFactors,
                                                                                           _state.gridConstraints,
                                                                                           MATCHER_DISTANCE_THRESHOLD_HIGH,
                                                                                           0.8f,
                                                                                           &currentFrame,
                                                                                           threshold);
                            printf("Found %i mapPoints by projection of local map\n", localMatches);
                        }
                    }

                    // TODO(jan): we call this function twice... is this necessary?
                    optimizePose(_state.orbExtractionParameters.inverseSigmaSquared,
                                 _state.fx,
                                 _state.fy,
                                 _state.cx,
                                 _state.cy,
                                 &currentFrame);

                    for (i32 i = 0; i < currentFrame.numberOfKeyPoints; i++)
                    {
                        MapPoint* mapPoint = currentFrame.mapPointMatches[i];
                        if (mapPoint)
                        {
                            if (!currentFrame.mapPointIsOutlier[i])
                            {
                                mapPoint->foundInKeyFrameCounter++;

                                if (mapPoint->observations.size() > 0)
                                {
                                    inlierMatches++;
                                }
                            }
                        }
                    }

                    // TODO(jan): reactivate this
                    // Decide if the tracking was succesful
                    // More restrictive if there was a relocalization recently
                    if (currentFrame.id < _state.lastRelocalizationFrameId + _state.maxFramesBetweenKeyFrames && inlierMatches < 50)
                    {
                        trackingIsOk = false;
                    }
                    else if (inlierMatches < 30)
                    {
                        trackingIsOk = false;
                    }
                    else
                    {
                        trackingIsOk = true;
                    }
                }
            }

            _state.trackingWasOk = trackingIsOk;

            if (trackingIsOk)
            {
                // TODO(jan): update motion model

                _pose = currentFrame.cTw.clone();

                // Clean VO matches
                for (int i = 0; i < currentFrame.numberOfKeyPoints; i++)
                {
                    MapPoint* mapPoint = currentFrame.mapPointMatches[i];
                    if (mapPoint)
                    {
                        if (mapPoint->observations.size() < 1)
                        {
                            currentFrame.mapPointMatches[i]   = nullptr;
                            currentFrame.mapPointIsOutlier[i] = false;
                        }
                    }
                }

                // TODO(jan): delete temporal map points (needed in motion model)

                bool32 addNewKeyFrame = needNewKeyFrame(currentFrame.id,
                                                        _state.lastKeyFrameId,
                                                        _state.lastRelocalizationFrameId,
                                                        _state.minFramesBetweenKeyFrames,
                                                        _state.maxFramesBetweenKeyFrames,
                                                        inlierMatches,
                                                        _state.referenceKeyFrame,
                                                        _state.keyFrames.size());

                if (addNewKeyFrame)
                {
                    KeyFrame* keyFrame = new KeyFrame();
                    initializeKeyFrame(keyFrame, currentFrame, _state.nextKeyFrameId);

                    _state.referenceKeyFrame       = keyFrame;
                    currentFrame.referenceKeyFrame = keyFrame;

                    _state.localMapping.newKeyFrames.push_back(keyFrame);

                    _state.lastKeyFrameId = keyFrame->index;

                    runLocalMapping(&_state.localMapping,
                                    _state.keyFrames.size(),
                                    _state.mapPoints,
                                    _state.keyFrames,
                                    _state.orbExtractionParameters,
                                    _state.fx,
                                    _state.fy,
                                    _state.cx,
                                    _state.cy,
                                    _state.invfx,
                                    _state.invfy,
                                    cameraMat,
                                    _state.nextMapPointId,
                                    _state.orbVocabulary,
                                    _state.invertedKeyFrameFile);

                    // TODO(jan): camera trajectory stuff
                }
            }
        }
        break;
    }
}

std::vector<KeyFrame*> WAI::ModeOrbSlam2DataOriented::getKeyFrames()
{
    std::vector<KeyFrame*> result = std::vector<KeyFrame*>(_state.keyFrames.begin(), _state.keyFrames.end());

    return result;
}

std::vector<MapPoint*> WAI::ModeOrbSlam2DataOriented::getMapPoints()
{
    std::vector<MapPoint*> result = std::vector<MapPoint*>(_state.mapPoints.begin(), _state.mapPoints.end());

    return result;
}

std::vector<MapPoint*> WAI::ModeOrbSlam2DataOriented::getLocalMapPoints()
{
    std::vector<MapPoint*> result = std::vector<MapPoint*>(_state.localMapPoints.begin(), _state.localMapPoints.end());

    return result;
}

std::vector<MapPoint*> WAI::ModeOrbSlam2DataOriented::getMatchedMapPoints()
{
    std::vector<MapPoint*> result;

    for (int i = 0; i < _state.lastFrame.numberOfKeyPoints; i++)
    {
        if (_state.lastFrame.mapPointMatches[i])
        {
            if (!_state.lastFrame.mapPointIsOutlier[i])
            {
                if (_state.lastFrame.mapPointMatches[i]->observations.size() > 0)
                {
                    result.push_back(_state.lastFrame.mapPointMatches[i]);
                }
            }
        }
    }

    return result;
}

bool WAI::ModeOrbSlam2DataOriented::getPose(cv::Mat* pose)
{
    *pose = _pose;

    return true;
}
