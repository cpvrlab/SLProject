//#############################################################################
//  File:      SLCVKeyframeDB.h
//  Author:    Raúl Mur-Artal, Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SLCVKEYFRAMEDB_H
#define SLCVKEYFRAMEDB_H

#include <SLCamera.h>
#include <SLCVFrame.h>
#include <SLCVKeyFrame.h>

#include <mutex>

//-----------------------------------------------------------------------------
//! AR Keyframe database class
/*! 
*/
class SLCVKeyFrameDB
{
public:
    SLCVKeyFrameDB(const ORBVocabulary &voc);

    void add(SLCVKeyFrame* pKF);
    void erase(SLCVKeyFrame* pKF);

    void clear();

    // Loop Detection
    std::vector<SLCVKeyFrame *> DetectLoopCandidates(SLCVKeyFrame* pKF, float minScore);

    // Relocalization
    std::vector<SLCVKeyFrame*> DetectRelocalizationCandidates(SLCVFrame* F);

protected:
    // Associated vocabulary
    const ORBVocabulary* mpVoc;
    // Inverted file
    std::vector<list<SLCVKeyFrame*> > mvInvertedFile;

    // Mutex
    std::mutex mMutex;
};

#endif // !SLCVKEYFRAMEDB_H
