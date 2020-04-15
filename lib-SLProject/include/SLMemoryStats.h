//#############################################################################
//  File:      SLMemoryStats.h
//  Author:    Michael Goettlicher
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_MEMSTATS_H
#define SL_MEMSTATS_H

#include <SL.h>
#include <SLImGuiInfosMemoryStats.h>

//! Callback function typedef for ImGui build function
typedef void(SL_STDCALL cbMemoryStats)();

class SLMemoryStats
{
    friend SLImGuiInfosMemoryStats;

public:
    //!returns true if memory statistics are valid
    bool valid() { return (_cb != NULL); }
    //!update statistics by calling callback
    void updateValue();

    //!install plattform dependent callback function pointer
    void setCallback(cbMemoryStats* cb);

    void setValues(long freeMemoryRT, long totalMemoryRT, long maxMemoryRT, long availMemoryAM, long totalMemoryAM, long thresholdAM, bool lowMemoryAM);

private:
    //! callback function: has to be installed system dependent before anything will work
    cbMemoryStats* _cb = NULL;

    //Values from android runtime:
    //!amount of free memory in the Java Virtual Machine.
    long _freeMemoryRT = 0;
    /*!Returns the total amount of memory in the Java virtual machine.
    * The value returned by this method may vary over time, depending on
    * the host environment. */
    long _totalMemoryRT = 0;
    //!maximum amount of memory that the Java virtual machine will attempt to use.
    long _maxMemoryRT = 0;

    long _usedMemInMB;
    long _maxHeapSizeInMB;
    long _availHeapSizeInMB;

    //Values from ActivityManager:
    /*!The available memory on the system.  This number should not
    be considered absolute: due to the nature of the kernel, a significant
    portion of this memory is actually in use and needed for the overall
    system to run well. */
    long _availMemoryAM = 0;
    /*!The total memory accessible by the kernel.  This is basically the
    RAM size of the device, not including below-kernel fixed allocations
    like DMA buffers, RAM for the baseband CPU, etc. */
    long _totalMemoryAM = 0;
    /*! The threshold of {@link #availMem} at which we consider memory to be
    low and start killing background services and other non-extraneous
    processes. */
    long _thresholdAM;
    /*!Set to true if the system considers itself to currently be in a low
    memory situation. */
    bool _lowMemoryAM = false;

    double _availableMegs;
    double _percentAvail;
};

#endif //SL_MEMSTATS_H
