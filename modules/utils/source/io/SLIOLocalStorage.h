//#############################################################################
//  File:      SLIOLocalStorage.h
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLIOLOCALSTORAGE_H
#define SLPROJECT_SLIOLOCALSTORAGE_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
//! Collection of functions for accessing browser local storage
namespace SLIOLocalStorage
{
bool exists(std::string path);
}
//-----------------------------------------------------------------------------
//! SLIOStream implementation for reading from browser local storage
/*!
 * The constructor loads the file from local storage into memory using the
 * JavaScript API provided by the web browser.
 */
class SLIOReaderLocalStorage : public SLIOReaderMemory
{
public:
    SLIOReaderLocalStorage(std::string path);
    ~SLIOReaderLocalStorage();
};
//-----------------------------------------------------------------------------
//! SLIOStream implementation for writing to browser local storage
/*!
 * When calling flush, the memory is written from memory to local storage using
 * the JavaScript API provided by the web browser.
 */
class SLIOWriterLocalStorage : public SLIOWriterMemory
{
public:
    SLIOWriterLocalStorage(std::string path);
    void flush();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOLOCALSTORAGE_H
