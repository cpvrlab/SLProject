//#############################################################################
//  File:      GlobalTimer.h
//  Author:    Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef GLOBAL_TIMER_H
#define GLOBAL_TIMER_H

#include "HighResTimer.h"

class GlobalTimer
{
public:
    static void  timerStart();
    static float timeS();
    static float timeMS();

private:
    static HighResTimer _timer;
};

#endif //GLOBAL_TIMER_H
