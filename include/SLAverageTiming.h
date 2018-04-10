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


//!concatenation of average value and timer
/*!
Define a hierarchy by posV and posH which is used in ui to arrange the measurements.
The first found content with posV==0 is used as reference measurement for the percental value.
*/
struct SLAverageTimingBlock {
    SLAverageTimingBlock(SLint averageNumValues, SLstring name, SLint posV, SLint posH)
        : val(averageNumValues),
        name(name),
        posV(posV),
        posH(posH)
    {}
    SLAvgFloat val;
    SLstring name;
    SLTimer timer;
    SLint posV=0;
    SLint posH=0;
    SLint nCalls=0;
};

//-----------------------------------------------------------------------------
//! Singleton timing class for average measurement of different timing blocks in loops
/*!
Call start("name", posV, posH) to define a new timing block and start timing or start timing
of an existing block. Call stop("name") to finish measurement for this block.
Define a hierarchy by posV and posH which is used in ui to arrange the measurements.
The first found content with posV==0 is used as reference measurement for the percental value.
*/
class SLAverageTiming : public std::map<std::string, SLAverageTimingBlock*>
{
public:
    SLAverageTiming();
    ~SLAverageTiming();

    //!start timer for a new or existing block
    static void start(const std::string& name, SLint posV, SLint posH);
    //!stop timer for a running block with name
    static void stop(const std::string& name);
    //!get time for block with name
    static SLfloat getTime(const std::string& name);
    //!get time for multiple blocks with given names
    static SLfloat getTime(const std::vector<std::string>& names);
    //!get the number of values
    static SLint getNumValues(const std::string& name);

    //!singleton
    static SLAverageTiming& instance()
    {
        static SLAverageTiming timing;
        return timing;
    }
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

    //average numValues
    SLint _averageNumValues = 200;
    SLint _currentPosV = 0;
    SLint _currentPosH = 0;
};
//-----------------------------------------------------------------------------

#endif //SL_AVERAGE_TIMING