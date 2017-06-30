//#############################################################################
//  File:      SLCVRaulMurOrb.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVRAULMURORB_H
#define SLCVRAULMURORB_H

#include <SLCV.h>

//-----------------------------------------------------------------------------
//!???
/*!???
*/
class SLCVRaulMurOrb: public cv::Feature2D
{
public:
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

                    SLCVRaulMurOrb(int nfeatures, float scaleFactor, int nlevels,
                                   int iniThFAST, int minThFAST);

                    ~SLCVRaulMurOrb(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.

    void            detectAndCompute    (SLCVInputArray image, SLCVInputArray mask,
                                         SLCVVKeyPoint& keypoints,
                                         SLCVOutputArray descriptors, 
                                         bool useProvidedKeypoints);

    int inline      GetLevels(){return nlevels;}
    float inline    GetScaleFactor(){return (float)scaleFactor;}
    SLVfloat inline GetScaleFactors(){return mvScaleFactor;}
    SLVfloat inline GetInverseScaleFactors(){return mvInvScaleFactor;}
    SLVfloat inline GetScaleSigmaSquares(){return mvLevelSigma2;}
    SLVfloat inline GetInverseScaleSigmaSquares(){return mvInvLevelSigma2;}

    SLCVVMat        mvImagePyramid;

protected:
    void            ComputePyramid          (SLCVMat image);
    void            ComputeKeyPointsOctTree (SLCVVVKeyPoint& allKeypoints);
    SLCVVKeyPoint   DistributeOctTree       (const SLCVVKeyPoint& vToDistributeKeys, 
                                             const int &minX,
                                             const int &maxX, 
                                             const int &minY, 
                                             const int &maxY, 
                                             const int &nFeatures, 
                                             const int &level);
    SLCVVPoint      pattern;
    int             nfeatures;
    double          scaleFactor;
    int             nlevels;
    int             iniThFAST;
    int             minThFAST;
    SLVint          mnFeaturesPerLevel;
    SLVint          umax;
    SLVfloat        mvScaleFactor;
    SLVfloat        mvInvScaleFactor;
    SLVfloat        mvLevelSigma2;
    SLVfloat        mvInvLevelSigma2;
};
//----------------------------------------------------------------------------
#endif // SLCVRAULMURORB_H
