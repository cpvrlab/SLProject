//#############################################################################
//  File:      SLCVStateEstimator.h
//  Author:    Jan Dellsperger
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVSTATEESTIMATOR_H
#define SLCVSTATEESTIMATOR_H

#include <stdafx.h>

class SLCVStateEstimator
{
public:
  SLMat4f getPose();
  void updatePose(const SLMat4f& slMat);

private:
  SLMat4f _pose;
  std::mutex _poseLock;
};

#endif
