//#############################################################################
//  File:      AverageTiming.cpp
//  Authors:   Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AverageTiming.h>
#include <HighResTimer.h>
#include <Utils.h>
#include <algorithm>
#include <cstring>

namespace Utils
{
//-----------------------------------------------------------------------------
AverageTiming::AverageTiming()
{
}
//-----------------------------------------------------------------------------
AverageTiming::~AverageTiming()
{
    for (auto& block : *this)
    {
        if (block.second)
            delete block.second;
    }
}
//-----------------------------------------------------------------------------
//! start timer for a new or existing block
void AverageTiming::start(const std::string& name)
{
    AverageTiming::instance().doStart(name);
}
//-----------------------------------------------------------------------------
//! stop timer for a running block with name
void AverageTiming::stop(const std::string& name)
{
    AverageTiming::instance().doStop(name);
}
//-----------------------------------------------------------------------------
//! get time for block with name
float AverageTiming::getTime(const std::string& name)
{
    return AverageTiming::instance().doGetTime(name);
}
//-----------------------------------------------------------------------------
//! get time for multiple blocks with given names
float AverageTiming::getTime(const std::vector<std::string>& names)
{
    return AverageTiming::instance().doGetTime(names);
}
//-----------------------------------------------------------------------------
//! get timings formatted via string
void AverageTiming::getTimingMessage(char* m)
{
    AverageTiming::instance().doGetTimingMessage(m);
}
//-----------------------------------------------------------------------------
//! start timer for a new or existing block
void AverageTiming::doStart(const std::string& name)
{
    if (find(name) == end())
    {
        AverageTimingBlock* block = new AverageTimingBlock(
          _averageNumValues, name, this->_currentPosV++, this->_currentPosH);
        (*this)[name] = block;
    }

    // if ((*this)[name]->isStarted)
    //     SL_LOG("AverageTiming: Block with name %s started twice!", name.c_str());

    (*this)[name]->timer.start();
    (*this)[name]->isStarted = true;

    this->_currentPosH++;
}

//-----------------------------------------------------------------------------
//! stop timer for a running block with name
void AverageTiming::doStop(const std::string& name)
{
    if (find(name) != end())
    {
        if (!(*this)[name]->isStarted)
            Utils::log("AverageTiming: Block with name %s stopped without being started!", name.c_str());
        (*this)[name]->timer.stop();
        (*this)[name]->val.set((*this)[name]->timer.elapsedTimeInMilliSec());
        (*this)[name]->nCalls++;
        (*this)[name]->isStarted = false;
        this->_currentPosH--;
    }
    else
        Utils::log("AverageTiming: A block with name %s does not exist!", name.c_str());
}

//-----------------------------------------------------------------------------
//! get time for block with name
float AverageTiming::doGetTime(const std::string& name)
{
    if (find(name) != end())
    {
        return (*this)[name]->val.average();
    }
    else
        Utils::log("AverageTiming: A block with name %s does not exist!", name.c_str());

    return 0.0f;
}

//-----------------------------------------------------------------------------
//! get time for multiple blocks with given names
float AverageTiming::doGetTime(const std::vector<std::string>& names) const
{
    AvgFloat val(_averageNumValues, 0.0f);
    for (const std::string& n : names)
    {
        val.set(getTime(n));
    }

    return val.average();
}
//-----------------------------------------------------------------------------
//! do get timings formatted via string
void AverageTiming::doGetTimingMessage(char* m)
{
    // sort vertically
    std::vector<AverageTimingBlock*> blocks;
    for (auto& block : AverageTiming::instance())
    {
        blocks.push_back(block.second);
    }
    std::sort(blocks.begin(), blocks.end(), [](AverageTimingBlock* lhs, AverageTimingBlock* rhs) -> bool
              { return lhs->posV < rhs->posV; });

    // find reference time
    float refTime = 1.0f;
    if (!blocks.empty())
    {
        refTime = (*blocks.begin())->val.average();
        // insert number of measurement calls
        snprintf(m + strlen(m), sizeof(m), "Num. calls: %i\n", (int)(*blocks.begin())->nCalls);
    }

    // calculate longest blockname
    size_t maxLen = 0;
    for (auto* block : blocks)
        if (block->name.length() > maxLen)
            maxLen = block->name.length();

    // insert time measurements
    for (auto* block : blocks)
    {
        float  val   = block->val.average();
        float  valPC = Utils::clamp(val / refTime * 100.0f, 0.0f, 100.0f);
        string name  = block->name;

        name.append(maxLen - name.length(), ' ');

        stringstream ss;
        // for (int i = 0; i < block->posH; ++i)
        //     ss << " ";
        ss << "%s: %4.1f ms (%3d%%)\n";
        snprintf(m + strlen(m), sizeof(m), ss.str().c_str(), name.c_str(), val, (int)valPC);
    }
} //-----------------------------------------------------------------------------
};