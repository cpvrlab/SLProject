//#############################################################################
//  File:      SLIOFetch.h
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLIOFETCH_H
#define SLPROJECT_SLIOFETCH_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//! SLIOStream implementation for downloading files from a web server
/*!
 * The constructor downloads the file via HTTP and stores it in memory. When
 * downloading, a loading screen is displayed to the user because it blocks
 * the entire application.
 */
//-----------------------------------------------------------------------------
class SLIOReaderFetch : public SLIOReaderMemory
{
public:
    static bool exists(std::string url);

    SLIOReaderFetch(std::string url);
    ~SLIOReaderFetch();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOFETCH_H
