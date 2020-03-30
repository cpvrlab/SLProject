//#############################################################################
//  File:      CVTrackedWAI.h
//  Author:    Michael Goettlicher, Marcus Hudritsch, Jan Dellsperger
//  Date:      Spring 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTrackedWAI_H
#define CVTrackedWAI_H

#include <CVTracked.h>

#include <WAISlam.h>

class CVTrackedWAI : public CVTracked
{
public:
    explicit CVTrackedWAI(std::string vocabularyFile);
    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib) final;

private:
    WAISlam*                 _mode              = nullptr;
    ORB_SLAM2::ORBextractor* _trackingExtractor = nullptr;
    ORBVocabulary*           _voc               = nullptr;
};

#endif