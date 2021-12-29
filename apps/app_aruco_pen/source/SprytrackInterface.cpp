//#############################################################################
//  File:      SprytrackInterface.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SprytrackInterface.h>

void SprytrackInterface::init()
{
    library = ftkInit();
    if (!library)
    {
        SL_EXIT_MSG("Failed to initialize the spryTrack library");
    }

    SL_LOG("SpryTrack: Initialized");
}

void SprytrackInterface::uninit()
{
    if (ftkClose(&library) != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("Failed to close the spryTrack library");
    }

    SL_LOG("SpryTrack: Uninitialized");
}