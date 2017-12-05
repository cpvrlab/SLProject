//#############################################################################
//  File:      SLCVKeyframeDB.h
//  Author:    Raúl Mur-Artal, Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVKEYFRAMEDB_H
#define SLCVKEYFRAMEDB_H

#include <SLCamera.h>
#include <SLCVFrame.h>
#include <SLCVKeyFrame.h>

//-----------------------------------------------------------------------------
//! AR Keyframe database class
/*! 
*/
class SLCVKeyFrameDB
{
public:
    SLCVKeyFrameDB(const ORBVocabulary &voc);

    SLCVVKeyFrame& keyFrames() { return _keyFrames; }

    void add(SLCVKeyFrame* pKF);

    // Relocalization
    std::vector<SLCVKeyFrame*> DetectRelocalizationCandidates(SLCVFrame* F);

protected:
    // Associated vocabulary
    const ORBVocabulary* mpVoc;
    // Inverted file
    std::vector<list<SLCVKeyFrame*> > mvInvertedFile;

private:
    SLCVVKeyFrame _keyFrames;
};

#endif // !SLCVKEYFRAMEDB_H
