//#############################################################################
//  File:      SLCVStateEstimator.cpp
//  Author:    Jan Dellsperger
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVStateEstimator.h>

SLMat4f SLCVStateEstimator::getPose()
{
  SLMat4f result;
  
  {
    std::lock_guard<std::mutex> guard(_poseLock);
    result = _pose;
  }
  
  return result;
}

void SLCVStateEstimator::updatePose(const SLMat4f& slMat)
{
  std::lock_guard<std::mutex> guard(_poseLock);
  _pose = slMat;
}
