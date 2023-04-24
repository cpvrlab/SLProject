//#############################################################################
//  File:      CVRaulMurOrb.h
//  Purpose:   Declares the Raul Mur ORB feature detector and descriptor
//  Source:    This File is based on the ORB Implementation of ORB_SLAM
//             https://github.com/raulmur/ORB_SLAM2
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Pascal Zingg, Timon Tschanz, Michael Goettlicher, Marcus Hudritsch
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVRAULMURORB_H
#define CVRAULMURORB_H

#include <CVTypedefs.h>

//-----------------------------------------------------------------------------
//! Orb detector and descriptor with distribution
class CVRaulMurOrb : public CVFeature2D
{
public:
    enum
    {
        HARRIS_SCORE = 0,
        FAST_SCORE   = 1
    };

    CVRaulMurOrb(int   nfeatures,
                 float scaleFactor,
                 int   nlevels,
                 int   iniThFAST,
                 int   minThFAST);

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void detectAndCompute(CVInputArray  image,
                          CVInputArray  mask,
                          CVVKeyPoint&  keypoints,
                          CVOutputArray descriptors,
                          bool          useProvidedKeypoints);

    uint          GetLevels() { return nlevels; }
    float         GetScaleFactor() { return (float)scaleFactor; }
    vector<float> GetScaleFactors() { return mvScaleFactor; }
    vector<float> GetInverseScaleFactors() { return mvInvScaleFactor; }
    vector<float> GetScaleSigmaSquares() { return mvLevelSigma2; }
    vector<float> GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

    CVVMat mvImagePyramid;

protected:
    void          ComputePyramid(CVMat image);
    void          ComputeKeyPointsOctTree(CVVVKeyPoint& allKeypoints);
    CVVKeyPoint   DistributeOctTree(const CVVKeyPoint& vToDistributeKeys,
                                    const int&         minX,
                                    const int&         maxX,
                                    const int&         minY,
                                    const int&         maxY,
                                    const int&         nFeatures,
                                    const int&         level);
    CVVPoint      pattern;
    int           nfeatures;
    double        scaleFactor;
    uint          nlevels;
    int           iniThFAST;
    int           minThFAST;
    vector<int>   mnFeaturesPerLevel;
    vector<int>   umax;
    vector<float> mvScaleFactor;
    vector<float> mvInvScaleFactor;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;
};
//----------------------------------------------------------------------------
#endif // CVRAULMURORB_H
