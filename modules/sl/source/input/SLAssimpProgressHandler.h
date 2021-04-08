//#############################################################################
//  File:      SLAssimpProgressHandler.h
//  Author:    Marcus Hudritsch
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLASSIMPPROGRESSHANDLER_H
#define SLASSIMPPROGRESSHANDLER_H

#include <assimp/ProgressHandler.hpp>
#include <AppDemo.h>

//-----------------------------------------------------------------------------
//!
class SLProgressHandler
{
public:
    virtual bool Update(float percentage = -1.f) = 0;
};
//-----------------------------------------------------------------------------
//!
class SLAssimpProgressHandler : SLProgressHandler, Assimp::ProgressHandler
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
#endif // SLASSIMPPROGRESSHANDLER_H
