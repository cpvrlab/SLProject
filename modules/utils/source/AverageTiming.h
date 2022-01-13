//#############################################################################
//  File:      AverageTiming.h
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef AVERAGE_TIMING
#define AVERAGE_TIMING

#include <string>
#include <map>

#include <HighResTimer.h>
#include <Averaged.h>
#include <sstream>
#include <utility>

namespace Utils
{
//! concatenation of average value and timer
/*!
Define a hierarchy by posV and posH which is used in ui to arrange the measurements.
The first found content with posV==0 is used as reference measurement for the percental value.
*/
struct AverageTimingBlock
{
    AverageTimingBlock(int averageNumValues, std::string name, int posV, int posH)
      : val(averageNumValues, 0.0f),
        name(std::move(name)),
        posV(posV),
        posH(posH)
    {
    }
    AvgFloat     val;
    std::string  name;
    HighResTimer timer;
    int          posV      = 0;
    int          posH      = 0;
    int          nCalls    = 0;
    bool         isStarted = false;
};

//-----------------------------------------------------------------------------
//! Singleton timing class for average measurement of different timing blocks in loops
/*!
Call start("name") to define a new timing block and start timing or start timing
of an existing block. Call stop("name") to finish measurement for this block.
Define a hierarchy by posV and posH which is used in ui to arrange the measurements.
The first found content with posV==0 is used as reference measurement for the percental value.
*/
class AverageTiming : public std::map<std::string, AverageTimingBlock*>
{
public:
    AverageTiming();
    ~AverageTiming();

    //! start timer for a new or existing block
    static void start(const std::string& name);
    //! stop timer for a running block with name
    static void stop(const std::string& name);
    //! get time for block with name
    static float getTime(const std::string& name);
    //! get time for multiple blocks with given names
    static float getTime(const std::vector<std::string>& names);
    //! get timings formatted via string
    static void getTimingMessage(char* m);

    //! singleton
    static AverageTiming& instance()
    {
        static AverageTiming timing;
        return timing;
    }

private:
    //! do start timer for a new or existing block
    void doStart(const std::string& name);
    //! do stop timer for a running block with name
    void doStop(const std::string& name);
    //! do get time for block with name
    float doGetTime(const std::string& name);
    //! do get time for multiple blocks with given names
    float doGetTime(const std::vector<std::string>& names) const;
    //! do get timings formatted via string
    void doGetTimingMessage(char* m);

    // average numValues
    int _averageNumValues = 200;
    int _currentPosV      = 0;
    int _currentPosH      = 0;
};

#define AVERAGE_TIMING_START(name) Utils::AverageTiming::start(name)
#define AVERAGE_TIMING_STOP(name) Utils::AverageTiming::stop(name)
//#define AVERAGE_TIMING_START
//#define AVERAGE_TIMING_STOP
//-----------------------------------------------------------------------------
};
#endif // AVERAGE_TIMING
