#include "WAIOrbExtraction.h"

static void initializeOrbExtractionParameters(OrbExtractionParameters* orbExtractionParameters,
                                              i32                      numberOfFeatures,
                                              i32                      numberOfScaleLevels,
                                              i32                      initialFastThreshold,
                                              i32                      minimalFastThreshold,
                                              r32                      scaleFactor,
                                              i32                      orbPatchSize,
                                              i32                      orbHalfPatchSize,
                                              i32                      edgeThreshold)
{
    *orbExtractionParameters = {};

    orbExtractionParameters->numberOfFeatures        = numberOfFeatures;
    orbExtractionParameters->numberOfScaleLevels     = numberOfScaleLevels;
    orbExtractionParameters->initialThreshold        = initialFastThreshold;
    orbExtractionParameters->minimalThreshold        = minimalFastThreshold;
    orbExtractionParameters->scaleFactor             = scaleFactor;
    orbExtractionParameters->logScaleFactor          = log(scaleFactor);
    orbExtractionParameters->orbOctTreeHalfPatchSize = orbHalfPatchSize;
    orbExtractionParameters->orbOctTreePatchSize     = orbPatchSize;
    orbExtractionParameters->edgeThreshold           = edgeThreshold;

    orbExtractionParameters->scaleFactors.resize(numberOfScaleLevels);
    orbExtractionParameters->sigmaSquared.resize(numberOfScaleLevels);
    orbExtractionParameters->scaleFactors[0] = 1.0f;
    orbExtractionParameters->sigmaSquared[0] = 1.0f;

    for (i32 i = 1; i < numberOfScaleLevels; i++)
    {
        orbExtractionParameters->scaleFactors[i] = (r32)(orbExtractionParameters->scaleFactors[i - 1] * scaleFactor);
        orbExtractionParameters->sigmaSquared[i] = (r32)(orbExtractionParameters->scaleFactors[i] * orbExtractionParameters->scaleFactors[i]);
    }

    orbExtractionParameters->inverseScaleFactors.resize(numberOfScaleLevels);
    orbExtractionParameters->inverseSigmaSquared.resize(numberOfScaleLevels);

    for (i32 i = 0; i < numberOfScaleLevels; i++)
    {
        orbExtractionParameters->inverseScaleFactors[i] = 1.0f / orbExtractionParameters->scaleFactors[i];
        orbExtractionParameters->inverseSigmaSquared[i] = 1.0f / orbExtractionParameters->sigmaSquared[i];
    }

    orbExtractionParameters->numberOfFeaturesPerScaleLevel.resize(numberOfScaleLevels);
    r32 factor                          = 1.0f / scaleFactor;
    r32 numberOfDesiredFeaturesPerScale = numberOfFeatures * (1 - factor) / (1 - (r32)pow((r64)factor, (r64)numberOfScaleLevels));

    i32 sumFeatures = 0;
    for (i32 level = 0; level < numberOfScaleLevels - 1; level++)
    {
        i32 featuresForLevel                                          = cvRound(numberOfDesiredFeaturesPerScale);
        orbExtractionParameters->numberOfFeaturesPerScaleLevel[level] = featuresForLevel;
        sumFeatures += featuresForLevel;
        numberOfDesiredFeaturesPerScale *= factor;
    }
    orbExtractionParameters->numberOfFeaturesPerScaleLevel[numberOfScaleLevels - 1] = std::max(numberOfFeatures - sumFeatures, 0);

    const int        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(orbExtractionParameters->orbPattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    orbExtractionParameters->umax.resize(orbHalfPatchSize + 1);

    int          v, v0, vmax = cvFloor(orbHalfPatchSize * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(orbHalfPatchSize * sqrt(2.f) / 2);
    const double hp2  = orbHalfPatchSize * orbHalfPatchSize;
    for (v = 0; v <= vmax; v++)
    {
        orbExtractionParameters->umax[v] = cvRound(sqrt(hp2 - v * v));
    }

    // Make sure we are symmetric
    for (v = orbHalfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (orbExtractionParameters->umax[v0] == orbExtractionParameters->umax[v0 + 1])
        {
            v0++;
        }
        orbExtractionParameters->umax[v] = v0;
        v0++;
    }
}

const r32   factorPI = (r32)(CV_PI / 180.f);
static void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat&      img,
                                 const cv::Point*    pattern,
                                 uchar*              desc)
{
    r32 angle = (r32)kpt.angle * factorPI;
    r32 a = (r32)cos(angle), b = (r32)sin(angle);

    const u8* center = &img.at<u8>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const i32 step   = (i32)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (i32 i = 0; i < 32; ++i, pattern += 16)
    {
        i32 t0, t1, val;
        t0  = GET_VALUE(0);
        t1  = GET_VALUE(1);
        val = t0 < t1;
        t0  = GET_VALUE(2);
        t1  = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

static r32 computeKeyPointAngle(const cv::Mat&          image,
                                cv::Point2f             pt,
                                const std::vector<i32>& umax,
                                const i32               halfPatchSize)
{
    i32 m_01 = 0, m_10 = 0;

    const u8* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (i32 u = -halfPatchSize; u <= halfPatchSize; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    i32 step = (i32)image.step1();
    for (i32 v = 1; v <= halfPatchSize; ++v)
    {
        // Proceed over the two lines
        i32 v_sum = 0;
        i32 d     = umax[v];
        for (i32 u = -d; u <= d; ++u)
        {
            i32 val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    r32 result = cv::fastAtan2((r32)m_01, (r32)m_10);

    return result;
}

void divideOrbExtractorNode(OrbExtractorNode& parentNode,
                            OrbExtractorNode& n1,
                            OrbExtractorNode& n2,
                            OrbExtractorNode& n3,
                            OrbExtractorNode& n4)
{
    const i32 halfX = ceil(static_cast<r32>(parentNode.topRight.x - parentNode.topLeft.x) / 2);
    const i32 halfY = ceil(static_cast<r32>(parentNode.bottomRight.y - parentNode.topLeft.y) / 2);

    //Define boundaries of childs
    n1.topLeft     = parentNode.topLeft;
    n1.topRight    = cv::Point2i(parentNode.topLeft.x + halfX, parentNode.topLeft.y);
    n1.bottomLeft  = cv::Point2i(parentNode.topLeft.x, parentNode.topLeft.y + halfY);
    n1.bottomRight = cv::Point2i(parentNode.topLeft.x + halfX, parentNode.topLeft.y + halfY);
    n1.keys.reserve(parentNode.keys.size());

    n2.topLeft     = n1.topRight;
    n2.topRight    = parentNode.topRight;
    n2.bottomLeft  = n1.bottomRight;
    n2.bottomRight = cv::Point2i(parentNode.topRight.x, parentNode.topLeft.y + halfY);
    n2.keys.reserve(parentNode.keys.size());

    n3.topLeft     = n1.bottomLeft;
    n3.topRight    = n1.bottomRight;
    n3.bottomLeft  = parentNode.bottomLeft;
    n3.bottomRight = cv::Point2i(n1.bottomRight.x, parentNode.bottomLeft.y);
    n3.keys.reserve(parentNode.keys.size());

    n4.topLeft     = n3.topRight;
    n4.topRight    = n2.bottomRight;
    n4.bottomLeft  = n3.bottomRight;
    n4.bottomRight = parentNode.bottomRight;
    n4.keys.reserve(parentNode.keys.size());

    //Associate points to childs
    for (size_t i = 0; i < parentNode.keys.size(); i++)
    {
        const cv::KeyPoint& kp = parentNode.keys[i];
        if (kp.pt.x < n1.topRight.x)
        {
            if (kp.pt.y < n1.bottomRight.y)
                n1.keys.push_back(kp);
            else
                n3.keys.push_back(kp);
        }
        else if (kp.pt.y < n1.bottomRight.y)
            n2.keys.push_back(kp);
        else
            n4.keys.push_back(kp);
    }

    if (n1.keys.size() == 1)
        n1.noMoreSubdivision = true;
    if (n2.keys.size() == 1)
        n2.noMoreSubdivision = true;
    if (n3.keys.size() == 1)
        n3.noMoreSubdivision = true;
    if (n4.keys.size() == 1)
        n4.noMoreSubdivision = true;
}

std::vector<cv::KeyPoint> distributeOctTree(const std::vector<cv::KeyPoint>& keyPointsToDistribute,
                                            const i32                        minX,
                                            const i32                        maxX,
                                            const i32                        minY,
                                            const i32                        maxY,
                                            const i32                        requiredFeatureCount,
                                            const i32                        level)
{
    // Compute how many initial nodes
    const r32 regionWidth  = (r32)(maxX - minX);
    const r32 regionHeight = (r32)(maxY - minY);

    const i32 initialNodeCount = std::round(regionWidth / regionHeight);

    const r32 initialNodeWidth = regionWidth / initialNodeCount;

    std::list<OrbExtractorNode> nodes;

    std::vector<OrbExtractorNode*> initialNodes;
    initialNodes.resize(initialNodeCount);

    for (i32 i = 0; i < initialNodeCount; i++)
    {
        OrbExtractorNode node = {};

        r32 leftX  = initialNodeWidth * static_cast<r32>(i);
        r32 rightX = initialNodeWidth * static_cast<r32>(i + 1);

        node.topLeft     = cv::Point2i(leftX, 0);
        node.topRight    = cv::Point2i(rightX, 0);
        node.bottomLeft  = cv::Point2i(leftX, regionHeight);
        node.bottomRight = cv::Point2i(rightX, regionHeight);
        node.keys.reserve(keyPointsToDistribute.size());

        nodes.push_back(node);
        initialNodes[i] = &nodes.back();
    }

    // Assign keypoints to initial nodes
    for (size_t i = 0; i < keyPointsToDistribute.size(); i++)
    {
        const cv::KeyPoint& kp = keyPointsToDistribute[i];
        initialNodes[kp.pt.x / initialNodeWidth]->keys.push_back(kp);
    }

    std::list<OrbExtractorNode>::iterator nodeIterator = nodes.begin();

    // flag, delete or leave initial nodes, according to their keypoint count
    while (nodeIterator != nodes.end())
    {
        if (nodeIterator->keys.size() == 1)
        {
            nodeIterator->noMoreSubdivision = true;
            nodeIterator++;
        }
        else if (nodeIterator->keys.empty())
        {
            nodeIterator = nodes.erase(nodeIterator);
        }
        else
        {
            nodeIterator++;
        }
    }

    bool32 finish = false;

    i32 iteration = 0;

    std::vector<std::pair<i32, OrbExtractorNode*>> nodesToExpand;
    nodesToExpand.reserve(nodes.size() * 4);

    while (!finish)
    {
        iteration++;

        i32 prevNodeCount = nodes.size();

        nodeIterator = nodes.begin();

        i32 amountOfNodesToExpand = 0;

        nodesToExpand.clear();

        while (nodeIterator != nodes.end())
        {
            if (nodeIterator->noMoreSubdivision)
            {
                // If node only contains one point do not subdivide and continue
                nodeIterator++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                OrbExtractorNode n1 = {}, n2 = {}, n3 = {}, n4 = {};
                divideOrbExtractorNode(*nodeIterator, n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.keys.size() > 0)
                {
                    nodes.push_front(n1);
                    if (n1.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n1.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n2.keys.size() > 0)
                {
                    nodes.push_front(n2);
                    if (n2.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n2.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n3.keys.size() > 0)
                {
                    nodes.push_front(n3);
                    if (n3.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n3.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n4.keys.size() > 0)
                {
                    nodes.push_front(n4);
                    if (n4.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n4.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }

                nodeIterator = nodes.erase(nodeIterator);
                //continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((i32)nodes.size() >= requiredFeatureCount || (i32)nodes.size() == prevNodeCount)
        {
            finish = true;
        }
        // continue dividing nodes until we have enough nodes to reach the required feature count
        else if (((i32)nodes.size() + amountOfNodesToExpand * 3) > requiredFeatureCount)
        {
            while (!finish)
            {
                prevNodeCount = (i32)nodes.size();

                std::vector<std::pair<i32, OrbExtractorNode*>> previousNodesToExpand = nodesToExpand;
                nodesToExpand.clear();

                std::sort(previousNodesToExpand.begin(), previousNodesToExpand.end());
                for (i32 j = previousNodesToExpand.size() - 1; j >= 0; j--)
                {
                    OrbExtractorNode n1 = {}, n2 = {}, n3 = {}, n4 = {};
                    divideOrbExtractorNode(*previousNodesToExpand[j].second, n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.keys.size() > 0)
                    {
                        nodes.push_front(n1);
                        if (n1.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n1.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n2.keys.size() > 0)
                    {
                        nodes.push_front(n2);
                        if (n2.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n2.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n3.keys.size() > 0)
                    {
                        nodes.push_front(n3);
                        if (n3.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n3.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n4.keys.size() > 0)
                    {
                        nodes.push_front(n4);
                        if (n4.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n4.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }

                    nodes.erase(previousNodesToExpand[j].second->iteratorToNode);

                    if ((i32)nodes.size() >= requiredFeatureCount) break;
                }

                if ((i32)nodes.size() >= requiredFeatureCount || (i32)nodes.size() == prevNodeCount)
                {
                    finish = true;
                }
            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> result;
    result.reserve(nodes.size());
    for (std::list<OrbExtractorNode>::iterator nodeIterator = nodes.begin();
         nodeIterator != nodes.end();
         nodeIterator++)
    {
        std::vector<cv::KeyPoint>& vNodeKeys   = nodeIterator->keys;
        cv::KeyPoint*              pKP         = &vNodeKeys[0];
        r32                        maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP         = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        result.push_back(*pKP);
    }

    return result;
}

static void computeScalePyramid(const cv::Mat           image,
                                const i32               numberOfScaleLevels,
                                const std::vector<r32>& inverseScaleFactors,
                                const i32               edgeThreshold,
                                std::vector<cv::Mat>&   imagePyramid)
{
    for (i32 level = 0; level < numberOfScaleLevels; level++)
    {
        r32      scale = inverseScaleFactors[level];
        cv::Size sz(cvRound((r32)image.cols * scale), cvRound((r32)image.rows * scale));
        cv::Size wholeSize(sz.width + edgeThreshold * 2, sz.height + edgeThreshold * 2);
        cv::Mat  temp(wholeSize, image.type()), masktemp;
        imagePyramid[level] = temp(cv::Rect(edgeThreshold, edgeThreshold, sz.width, sz.height));

        if (level)
        {
            resize(imagePyramid[level - 1],
                   imagePyramid[level],
                   sz,
                   0,
                   0,
                   CV_INTER_LINEAR);

            copyMakeBorder(imagePyramid[level],
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image,
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101);
        }
    }
}

/**
 * 1. Splits every level of the image into evenly sized cells
 * 2. Detects corners in a 7x7 cell area
 * 3. Make sure key points are well distributed
 * 4. Compute orientation of keypoints
 */
void computeKeyPointsInOctTree(const std::vector<cv::Mat>&             imagePyramid,
                               const i32                               numberOfScaleLevels,
                               const std::vector<r32>&                 scaleFactors,
                               const i32                               edgeThreshold,
                               const i32                               numberOfFeatures,
                               const std::vector<i32>&                 numberOfFeaturesPerScaleLevel,
                               const i32                               initialFastThreshold,
                               const i32                               minimalFastThreshold,
                               const i32                               patchSize,
                               const i32                               halfPatchSize,
                               const std::vector<i32>                  umax,
                               std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(numberOfScaleLevels);

    const r32 W = 30;

    for (i32 level = 0; level < numberOfScaleLevels; level++)
    {
        const i32 minBorderX = edgeThreshold - 3;
        const i32 minBorderY = minBorderX;
        const i32 maxBorderX = imagePyramid[level].cols - edgeThreshold + 3;
        const i32 maxBorderY = imagePyramid[level].rows - edgeThreshold + 3;

        std::vector<cv::KeyPoint> keyPointsToDistribute;
        keyPointsToDistribute.reserve(numberOfFeatures * 10);

        const r32 width  = (maxBorderX - minBorderX);
        const r32 height = (maxBorderY - minBorderY);

        const i32 nCols = width / W;
        const i32 nRows = height / W;
        const i32 wCell = ceil(width / nCols);
        const i32 hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float       maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                cv::FAST(imagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell,
                         initialFastThreshold,
                         true);

                if (vKeysCell.empty())
                {
                    cv::FAST(imagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                             vKeysCell,
                             minimalFastThreshold,
                             true);
                }

                if (!vKeysCell.empty())
                {
                    for (std::vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        keyPointsToDistribute.push_back(*vit);
                    }
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(numberOfFeatures);

        keypoints = distributeOctTree(keyPointsToDistribute,
                                      minBorderX,
                                      maxBorderX,
                                      minBorderY,
                                      maxBorderY,
                                      numberOfFeaturesPerScaleLevel[level],
                                      level);

        const i32 scaledPatchSize = patchSize * scaleFactors[level];

        // Add border to coordinates and scale information
        for (i32 i = 0; i < keypoints.size(); i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size   = (r32)scaledPatchSize;
        }
    }

    for (i32 level = 0; level < numberOfScaleLevels; level++)
    {
        std::vector<cv::KeyPoint>& keyPoints = allKeypoints[level];
        for (std::vector<cv::KeyPoint>::iterator keyPoint    = keyPoints.begin(),
                                                 keyPointEnd = keyPoints.end();
             keyPoint != keyPointEnd;
             keyPoint++)
        {
            keyPoint->angle = computeKeyPointAngle(imagePyramid[level],
                                                   keyPoint->pt,
                                                   umax,
                                                   halfPatchSize);
        }
    }
}
