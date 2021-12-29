//#############################################################################
//  File:      SprytrackInterface.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_SPRYTRACKINTERFACE_H
#define SRC_SPRYTRACKINTERFACE_H

#include <ftkInterface.h>
#include <SL.h>

class SprytrackInterface
{
public:
    static SprytrackInterface& instance()
    {
        static SprytrackInterface instance;
        return instance;
    }

private:
    SprytrackInterface() {}

public:
    void init();
    void uninit();

private:
    ftkLibrary library = nullptr;

};

#endif // SRC_SPRYTRACKINTERFACE_H
