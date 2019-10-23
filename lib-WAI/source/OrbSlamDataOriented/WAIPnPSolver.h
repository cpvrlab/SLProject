#ifndef WAI_PNPSOLVER_H
#define WAI_PNPSOLVER_H

struct RansacParameters
{
    r64 probability;
    i32 minInliers;
    i32 maxIterations;
    i32 minSet;
    r32 epsilon;
    r32 th2;
};

struct PnPSolver
{
    std::vector<MapPoint*>   mvpMapPointMatches;
    std::vector<cv::Point2f> mvP2D;
    std::vector<cv::Point3f> mvP3Dw;
    std::vector<size_t>      mvKeyPointIndices;
    std::vector<size_t>      mvAllIndices;
    std::vector<bool32>      mvbInliersi; // TODO(jan): what is this?
    std::vector<r32>         mvMaxError;
    std::vector<r32>         mvSigma2;

    r32  uc, vc, fu, fv;
    r64 *pws, *us, *alphas, *pcs;

    double mRi[3][3];
    double mti[3];

    r64 cws[4][3], ccs[4][3];
    r64 cwsDeterminant;

    i32 maximumNumberOfCorrespondences;
    i32 numberOfCorrespondences;

    i32 pointCount;
    i32 inliersCount;

    // Ransac state
    i32                 globalIterationCount;
    std::vector<bool32> mvbBestInliers; // TODO(jan): what is this?
    i32                 bestInliersCount;
    cv::Mat             mBestTcw;

    // Refined
    cv::Mat             mRefinedTcw;
    std::vector<bool32> mvbRefinedInliers;
    int                 mnRefinedInliers;

    RansacParameters ransacParameters;
};

#endif