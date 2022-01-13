//#############################################################################
//  File:      SLAssimpProgressHandler.h
//  Authors:   Marcus Hudritsch
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLASSIMPPROGRESSHANDLER_H
#define SLASSIMPPROGRESSHANDLER_H

#ifdef SL_BUILD_WITH_ASSIMP
#    include <assimp/ProgressHandler.hpp>
#    include <AppDemo.h>

//-----------------------------------------------------------------------------
//!
class SLProgressHandler
{
public:
    virtual bool Update(float percentage = -1.f) = 0;
};
//-----------------------------------------------------------------------------
//!
class SLAssimpProgressHandler : SLProgressHandler
  , Assimp::ProgressHandler
{
public:
    virtual bool Update(float percentage = -1.f)
    {
        if (percentage >= 0.0f && percentage <= 100.0f)
        {
            AppDemo::jobProgressNum((SLint)percentage);
            return true;
        }
        else
            return false;
    }
};
//-----------------------------------------------------------------------------
#endif // SL_BUILD_WITH_ASSIMP
#endif // SLASSIMPPROGRESSHANDLER_H
