//#############################################################################
//  File:      SLCVDescriptor.h
//  Author:    Marcus Hudritsch
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVDESCRIPTOR_H
#define SLCVDESCRIPTOR_H

#include <SLCV.h>

//-----------------------------------------------------------------------------
class SLCVDescriptor
{
private:
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
public:
                        SLCVDescriptor  (SLCVDescriptorType type);

    void                compute         (SLCVInputArray image,
                                         SLCVVKeyPoint  &keypoints,
                                         SLCVOutputArray descriptors);

    void                detectAndCompute(SLCVInputArray image,
                                         SLCVVKeyPoint &keypoints,
                                         SLCVOutputArray descriptors,
                                         SLCVInputArray mask=cv::noArray());

    SLCVDescriptorType  type;

    void setDescriptor(cv::Ptr<cv::DescriptorExtractor> descriptor) { _descriptor = descriptor; }
};
//-----------------------------------------------------------------------------
#endif // SLCVDESCRIPTOR_H
