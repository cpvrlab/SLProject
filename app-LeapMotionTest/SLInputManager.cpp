//#############################################################################
//  File:      SLInputManager.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLInputManager.h>


SLInputManager SLInputManager::_instance;

SLInputManager& SLInputManager::instance()
{
    return _instance;
}

SLInputManager::~SLInputManager()
{
    /// @todo   decide if all input devices have to be managed by the input manager or if someone
    ///         else has to manage their life time.
    //for(SLint i = 0; i < _devices.size(); ++i)
        //delete _devices[i];

    _devices.clear();
}

void SLInputManager::update()
{
    for(SLint i = 0; i < _devices.size(); ++i)
        _devices[i]->poll();
}