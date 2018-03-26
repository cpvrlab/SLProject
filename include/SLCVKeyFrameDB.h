//#############################################################################
//  File:      SLCVKeyframeDB.h
//  Author:    Raúl Mur-Artal, Michael Goettlicher
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
    ~SLCVKeyFrameDB();

    SLCVVKeyFrame& keyFrames() { return _keyFrames; }

    void add(SLCVKeyFrame* pKF);

    void clear();

    // Relocalization
    std::vector<SLCVKeyFrame*> DetectRelocalizationCandidates(SLCVFrame* F);

    //getters
    bool renderKfBackground() { 
        return _renderKfBackground; 
    }
    bool allowAsActiveCam() {
        return _allowAsActiveCam;
    }
    
    //setters
    void renderKfBackground(bool s) { 
        _renderKfBackground = s; 
    }
    void allowAsActiveCam(bool s) {
        _allowAsActiveCam = s;
    }
protected:
    // Associated vocabulary
    const ORBVocabulary* mpVoc;
    // Inverted file
    std::vector<list<SLCVKeyFrame*> > mvInvertedFile;

private:
    SLCVVKeyFrame _keyFrames;

    //if backgound rendering is active kf images will be rendered on 
    //near clipping plane if kf is not the active camera
    bool _renderKfBackground = false;
    //allow SLCVCameras as active camera so that we can look through it
    bool _allowAsActiveCam = false;
};

#endif // !SLCVKEYFRAMEDB_H
