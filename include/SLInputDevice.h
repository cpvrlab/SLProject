//#############################################################################
//  File:      SLInputDevice.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINPUTDEVICE_H
#define SLINPUTDEVICE_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Interface for input devices that have to be pollsed
class SLInputDevice
{
public:
    SLInputDevice();
    ~SLInputDevice();
    
    void            enable  ();
    void            disable ();
    virtual void    poll    () = 0;
};

typedef vector<SLInputDevice*> SLVInputDevice;

#endif