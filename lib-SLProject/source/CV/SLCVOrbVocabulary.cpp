//#############################################################################
//  File:      SLCVOrbVocabulary.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVOrbVocabulary.h>
#include <SLCVCalibration.h>


//-----------------------------------------------------------------------------
SLCVOrbVocabulary::~SLCVOrbVocabulary()
{
    if (_vocabulary)
        delete _vocabulary;
}
//-----------------------------------------------------------------------------
void SLCVOrbVocabulary::loadFromFile()
{
    _vocabulary = new ORB_SLAM2::ORBVocabulary();
    string strVocFile = SLCVCalibration::calibIniPath + "ORBvoc.bin";
    bool bVocLoad = _vocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
    {
        SL_LOG("Wrong path to vocabulary. Failed to open at: %s", strVocFile.c_str());
        SL_EXIT_MSG("SLCVOrbVocabulary::loadFromFile: failed to load vocabulary");
    }
    SL_LOG("Vocabulary loaded!\n");
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* SLCVOrbVocabulary::get()
{
    return instance().doGet();
}
//-----------------------------------------------------------------------------
void SLCVOrbVocabulary::free()
{
    instance().doFree();
}
//-----------------------------------------------------------------------------
void SLCVOrbVocabulary::doFree()
{
    if (_vocabulary)
        delete _vocabulary;
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* SLCVOrbVocabulary::doGet()
{
    if (!_vocabulary)
        loadFromFile();
    return _vocabulary;
}