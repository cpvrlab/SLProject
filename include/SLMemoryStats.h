//#############################################################################
//  File:      SLMemoryStats.h
//  Author:    Michael Goettlicher
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_MEMSTATS_H
#define SL_MEMSTATS_H

#include <SL.h>
#include <SLInterface.h>

//! Callback function typedef for ImGui build function
typedef void (SL_STDCALL cbMemoryStats)();

class SLMemoryStats
{
public:
    //!returns true if memory statistics are valid
    bool valid()
    {
        return (_cb != NULL && _updated);
    }
    //!update statistics by calling callback
    void updateValue()
    {
        //execute callback to 
        if (_cb)
            (*_cb)();
        else {
            SL_WARN_MSG("SLMemoryStats update called but no callback is installed!\n");
            _updated = false;
        }
    }
    //!install plattform dependent callback function pointer
    void setCallback(cbMemoryStats* cb)
    {
        _cb = cb;
    }

    void setValue(double val)
    {
        _val = val; 
    }

    double getValue()
    {
        return _val;
    }

private:
    double _val = 0.0;
    bool _updated = false;
    cbMemoryStats* _cb = NULL;
};


#endif //SL_MEMSTATS_H
