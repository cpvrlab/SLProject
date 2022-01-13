//#############################################################################
//  File:      sl/SLObject.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLREFOBJ_H
#define SLREFOBJ_H

#include <SL.h>

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
    SLObject(const SLstring& Name = "",
             const SLstring& url  = "")
    {
        _name = Name;
        _url  = url;
    }
    virtual ~SLObject() {}

    // Setters
    void name(const SLstring& Name) { _name = Name; }
    void url(const SLstring& url) { _url = url; }

    // Getters
    const SLstring& name() const { return _name; }
    const SLstring& url() const { return _url; }

protected:
    SLstring _name; //!< name of an object
    SLstring _url;  //!< uniform resource locator
};
//-----------------------------------------------------------------------------
#endif
