//#############################################################################
//  File:      SLAverageTiming.h
//  Author:    Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SL_AVERAGE_TIMING
#define SL_AVERAGE_TIMING

#include <string>
#include <map>

#include <SLTimer.h>
#include <SLAverage.h>
#include <sstream>

//-----------------------------------------------------------------------------
//! Singleton timing class for average measurement of different timing blocks
/*!
Call start("name") to define a new timing block and start timing or start timing
of an existing block. Call stop("name")
*/
class SLAverageTiming
{
    //!concatenation of average value and timer
    struct Block {
        SLAvgFloat val;
        SLTimer timer;
    };

public:
    SLAverageTiming();
    ~SLAverageTiming();

    //!start timer for a new or existing block
    static void start(const std::string& name);
    //!stop timer for a running block with name
    static void stop(const std::string& name);
    //!get time for block with name
    static SLfloat getTime(const std::string& name);
    //!get time for multiple blocks with given names
    static SLfloat getTime(const std::vector<std::string>& names);
    //!get the number of values
    static SLint getNumValues(const std::string& name);
private:
    //!do start timer for a new or existing block
    void doStart(const std::string& name);
    //!do stop timer for a running block with name
    void doStop(const std::string& name);
    //!do get time for block with name
    SLfloat doGetTime(const std::string& name);
    //!do get time for multiple blocks with given names
    SLfloat doGetTime(const std::vector<std::string>& names);
    //!do get the number of values
    SLint doGetNumValues(const std::string& name);

    //!singleton
    static SLAverageTiming& instance()
    {
        static SLAverageTiming timing;
        return timing;
    }

    //!time measurement blocks
    std::map<std::string, Block*> _blocks;
};
//-----------------------------------------------------------------------------

#endif //SL_AVERAGE_TIMING