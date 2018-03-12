//#############################################################################
//  File:      SLAverageTiming.cpp
//  Author:    Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#include <SLAverageTiming.h>

//-----------------------------------------------------------------------------
SLAverageTiming::SLAverageTiming()
{

}
//-----------------------------------------------------------------------------
//!start timer for a new or existing block
void SLAverageTiming::start(const std::string& name)
{
    SLAverageTiming::instance().doStart(name);
}
//-----------------------------------------------------------------------------
//!stop timer for a running block with name
void SLAverageTiming::stop(const std::string& name)
{
    SLAverageTiming::instance().doStop(name);
}
//-----------------------------------------------------------------------------
//!get time for block with name
SLfloat SLAverageTiming::getTime(const std::string& name)
{
    return SLAverageTiming::instance().getTime(name);
}
//-----------------------------------------------------------------------------
//!get time for multiple blocks with given names
SLfloat SLAverageTiming::getTime(const std::vector<std::string>& names)
{
    return SLAverageTiming::instance().doGetTime(names);
}

//-----------------------------------------------------------------------------
//!start timer for a new or existing block
void SLAverageTiming::doStart(const std::string& name)
{
    if (_blocks.find(name) == _blocks.end()) {
        _blocks[name] = Block();
    }

    _blocks[name].timer.start();
}

//-----------------------------------------------------------------------------
//!stop timer for a running block with name
void SLAverageTiming::doStop(const std::string& name)
{
    if (_blocks.find(name) != _blocks.end()) {
        _blocks[name].timer.stop();
        _blocks[name].val.set(_blocks[name].timer.elapsedTimeInMicroSec());
    }
    else
        SL_LOG("SLAverageTiming: A block with name %s does not exist!\n", name.c_str());
}

//-----------------------------------------------------------------------------
//!get time for block with name
SLfloat SLAverageTiming::doGetTime(const std::string& name)
{
    if (_blocks.find(name) != _blocks.end()) {
        return _blocks[name].val.average();
    }
    else
        SL_LOG("SLAverageTiming: A block with name %s does not exist!\n", name.c_str());

    return 0.0f;
}

//-----------------------------------------------------------------------------
//!get time for multiple blocks with given names
SLfloat SLAverageTiming::doGetTime(const std::vector<std::string>& names)
{
    SLAvgFloat val;
    for (const std::string& n : names)
    {
        val.set(getTime(n));
    }

    return val.average();
}

//-----------------------------------------------------------------------------