//#############################################################################
//  File:      SLMemoryStats.cpp
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLMemoryStats.h>

//-----------------------------------------------------------------------------
//!update statistics by calling callback
void SLMemoryStats::updateValue()
{
    //execute callback to
    if (_cb)
    {
        (*_cb)();
    }
    else
    {
        SL_WARN_MSG("SLMemoryStats update called but no callback is installed!\n");
    }
}
//-----------------------------------------------------------------------------
//!install plattform dependent callback function pointer
void SLMemoryStats::setCallback(cbMemoryStats* cb)
{
    _cb = cb;
}
//-----------------------------------------------------------------------------
void SLMemoryStats::setValues(long freeMemoryRT, long totalMemoryRT, long maxMemoryRT, long availMemoryAM, long totalMemoryAM, long thresholdAM, bool lowMemoryAM)
{
    _freeMemoryRT  = freeMemoryRT;
    _totalMemoryRT = totalMemoryRT;
    _maxMemoryRT   = maxMemoryRT;
    _availMemoryAM = availMemoryAM;
    _totalMemoryAM = totalMemoryAM;
    _thresholdAM   = thresholdAM;
    _lowMemoryAM   = lowMemoryAM;

    //calculate additional values:
    _usedMemInMB       = (_totalMemoryRT - _freeMemoryRT) / 1048576L;
    _maxHeapSizeInMB   = _maxMemoryRT / 1048576L;
    _availHeapSizeInMB = _maxHeapSizeInMB - _usedMemInMB;

    _availableMegs = _availMemoryAM / 0x100000L;
    _percentAvail  = _availMemoryAM / (double)_totalMemoryAM * 100.0;

    //Explanation of the number 0x100000L:
    //1024 bytes == 1 Kibibyte
    //1024 Kibibyte == 1 Mebibyte
    //1024 * 1024 == 1048576
    //1048576 == 0x100000
}