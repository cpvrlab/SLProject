//#############################################################################
//  File:      ByteOrder.h
//  Authors:   Marino von Wattenwyl
//  Date:      January 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_BYTEORDER_H
#define SLPROJECT_BYTEORDER_H

#include <cstdint>
#include <climits>
#include <cstdlib>
#include <iostream>

//-----------------------------------------------------------------------------
//! Abort compilation if a char has not 8 bits, as functions for this case aren't implemented yet
#if CHAR_BIT != 8
#    error "Byte order functions require that the machine has a char width of 8 bit"
#endif
//-----------------------------------------------------------------------------
//! Utility functions for functions related to byte order conversions
namespace ByteOrder
{
//-----------------------------------------------------------------------------
void toBigEndian16(uint16_t src, char* dest);
void toBigEndian32(uint32_t src, char* dest);
void toBigEndian64(uint64_t src, char* dest);
//-----------------------------------------------------------------------------
void writeBigEndian16(uint16_t number, std::ostream& stream);
void writeBigEndian32(uint32_t number, std::ostream& stream);
void writeBigEndian64(uint64_t number, std::ostream& stream);
//-----------------------------------------------------------------------------
} // namespace ByteOrder
//-----------------------------------------------------------------------------

#endif // SLPROJECT_BYTEORDER_H
