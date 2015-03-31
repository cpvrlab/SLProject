//#############################################################################
//  File:      SL/SLObject.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLREFOBJ_H
#define SLREFOBJ_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Base class for all other classes
/*!      
The SLObject class serves as root3D class for other classes and provides for the
moment only a string with the name. It could be extended for object i/o 
(serialization) or reference counting.
*/
class SLObject
{
    public:               
                            SLObject(const SLstring& Name="")
                            {  _name = Name;
                            }
            virtual         ~SLObject(){}
            
            // Setters
            void            name(const SLstring& Name){_name = Name;}
            
            // Getters
            const SLstring& name() const {return _name;}
   
    protected:
            SLstring        _name;    //!< name of an object
};
//-----------------------------------------------------------------------------
#endif
