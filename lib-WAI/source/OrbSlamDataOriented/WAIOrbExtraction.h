#ifndef WAI_ORB_EXTRACTION_H
#define WAI_ORB_EXTRACTION_H

#include <WAIPlatform.h>

#include <opencv2/core/core.hpp>
#include <list>

struct OrbExtractorNode
{
    std::vector<cv::KeyPoint>             keys;
    cv::Point2i                           topLeft, topRight, bottomLeft, bottomRight;
    std::list<OrbExtractorNode>::iterator iteratorToNode;
    bool32                                noMoreSubdivision;
};

struct OrbExtractionParameters
{
    i32 numberOfFeatures;
    r32 scaleFactor;
    r32 logScaleFactor;
    i32 numberOfScaleLevels;
    i32 initialThreshold;
    i32 minimalThreshold;

    std::vector<r32> scaleFactors;
    std::vector<r32> inverseScaleFactors;
    std::vector<r32> sigmaSquared;
    std::vector<r32> inverseSigmaSquared;

    std::vector<i32> numberOfFeaturesPerScaleLevel;

    std::vector<i32>       umax;
    std::vector<cv::Point> orbPattern;

    i32 edgeThreshold;
    i32 orbOctTreePatchSize;
    i32 orbOctTreeHalfPatchSize;
};

#endif