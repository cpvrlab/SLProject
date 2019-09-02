//#############################################################################
//  File:      WAIKeyframeDB.h
//  Author:    Ra�l Mur-Artal, Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Ra�l Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

#ifndef WAIKEYFRAMEDB_H
#define WAIKEYFRAMEDB_H

#include <WAIHelper.h>
#include <WAIKeyFrame.h>
#include <OrbSlam/ORBVocabulary.h>

#include <mutex>

//-----------------------------------------------------------------------------
//! AR Keyframe database class
/*! 
*/
class WAI_API WAIKeyFrameDB
{
    public:
    WAIKeyFrameDB(const ORB_SLAM2::ORBVocabulary& voc);

    void add(WAIKeyFrame* pKF);
    void erase(WAIKeyFrame* pKF);

    void clear();

    // Loop Detection
    enum LoopDetectionErrorCodes
    {
        LOOP_DETECTION_ERROR_NONE,
        LOOP_DETECTION_ERROR_NO_CANDIDATES_WITH_COMMON_WORDS,
        LOOP_DETECTION_ERROR_NO_SIMILAR_CANDIDATES
    };
    std::vector<WAIKeyFrame*> DetectLoopCandidates(WAIKeyFrame* pKF, float minScore, int* errorCode);

    // Relocalization
    std::vector<WAIKeyFrame*> DetectRelocalizationCandidates(WAIFrame* F);

    protected:
    // Associated vocabulary
    const ORBVocabulary* mpVoc;
    // Inverted file
    std::vector<list<WAIKeyFrame*>> mvInvertedFile;

    // Mutex
    std::mutex mMutex;
};

#endif // !WAIKEYFRAMEDB_H
