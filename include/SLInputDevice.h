//#############################################################################
//  File:      SLInputDevice.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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

    /** Polls a custom input device. returns true if the poll resulted in
    event's being sent out that were accepted by some receiver. */
    virtual SLbool  poll    () = 0;
};

typedef vector<SLInputDevice*> SLVInputDevice;

#endif