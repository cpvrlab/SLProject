//#############################################################################
//  File:      WAIOrbVocabulary.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIOrbVocabulary.h>
#include <Utils.h>

WAIOrbVocabulary::~WAIOrbVocabulary()
{
    if (_vocabulary)
        delete _vocabulary;
}
//-----------------------------------------------------------------------------
bool WAIOrbVocabulary::initialize(std::string filename)
{
    bool result = instance().doInitialize(filename);

    return result;
}
//-----------------------------------------------------------------------------
bool WAIOrbVocabulary::loadFromFile(std::string strVocFile)
{
    bool result = false;

    if (!_vocabulary)
    {
        _vocabulary = new ORB_SLAM2::ORBVocabulary();
    }

    bool bVocLoad = _vocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
    {
        Utils::log("WAI", "Wrong path to vocabulary. Failed to open at: %s", strVocFile.c_str());
        Utils::log("WAI", "WAIOrbVocabulary::loadFromFile: failed to load vocabulary");
    }
    else
    {
        result = true;
    }

    return result;
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* WAIOrbVocabulary::get()
{
    return instance().doGet();
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::free()
{
    instance().doFree();
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::doFree()
{
    if (_vocabulary)
        delete _vocabulary;

    _vocabulary = nullptr;
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* WAIOrbVocabulary::doGet()
{
    return _vocabulary;
}
//-----------------------------------------------------------------------------
bool WAIOrbVocabulary::doInitialize(std::string filename)
{
    bool result = loadFromFile(filename);

    return result;
}
