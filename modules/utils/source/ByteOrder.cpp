//#############################################################################
//  File:      ByteOrder.cpp
//  Authors:   Marino von Wattenwyl
//  Date:      January 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <ByteOrder.h>
#include <cstring>

//-----------------------------------------------------------------------------
namespace ByteOrder
{
//-----------------------------------------------------------------------------
/*! Converts a 16-bit number from little-endian to big-endian regardless of the host byte order.
 * See toBigEndian32() for an explanation of the algorithm.
 * @param src the 16-bit number that should be converted
 * @param dest the pointer where the big-endian result will be written to
 */
void toBigEndian16(uint16_t src, char* dest)
{
    char*    arr = (char*)&src;
    uint16_t res = (uint16_t)arr[0] << 8 |
                   (uint16_t)arr[1];
    memcpy(dest, &res, 2);
}
//-----------------------------------------------------------------------------
/*! Converts a 32-bit number from little-endian to big-endian regardless of the host byte order.
 * Here is an example that shows how the algorithm works:
 *
 * Let's say we want to convert the number 8, which has the following memory layouts:
 *
 * Little-endian: [8 0 0 0]
 * Big-Endian:    [0 0 0 8]
 *
 * Now we take the address of the number and convert it to a char pointer to
 * get access to the individual bytes of the number:
 *
 * char* arr = (char*)&src;
 *
 * This results in the following values for the array indices:
 *
 * Little-endian: [0] => 8, [1] => 0, [2] => 0, [3] => 0
 * Big-endian:    [0] => 0, [1] => 0, [2] => 0, [3] => 8
 *
 * Next, we take all the array elements and shift them by (24 - index) bytes to the left:
 *
 * uint32_t res =
 * (uint32_t)arr[0] << 24 |
 * (uint32_t)arr[1] << 16 |
 * (uint32_t)arr[2] <<  8 |
 * (uint32_t)arr[3] <<  0;
 *
 * On a little-endian system, the 8 will be shifted 24 bits to the left and be the most significant
 * byte, which is stored last in little-endian:
 *
 * [0 0 0 8]
 *
 * On a big-endian system, the 8 will not be shifted and remain the least significant byte,
 * which is also stored last in big-endian:
 *
 * [0 0 0 8]
 *
 * We have thus achieved a host independent conversion to big-endian
 * Finally, we copy the 4 bytes to the destination:
 *
 * std::memcpy(dest, &res, 4);
 *
 * @param src the 32-bit number that should be converted
 * @param dest the pointer where the big-endian result will be written to
 */
void toBigEndian32(uint32_t src, char* dest)
{
    char*    arr = (char*)&src;
    uint32_t res = (uint32_t)arr[0] << 24 |
                   (uint32_t)arr[1] << 16 |
                   (uint32_t)arr[2] << 8 |
                   (uint32_t)arr[3];
    memcpy(dest, &res, 4);
}
//-----------------------------------------------------------------------------
/*! Converts a 64-bit number from little-endian to big-endian regardless of the host byte order.
 * See toBigEndian32() for an explanation of the algorithm.
 * @param src the 64-bit number that should be converted
 * @param dest the pointer where the big-endian result will be written to
 */
void toBigEndian64(uint64_t src, char* dest)
{
    char*    arr = (char*)&src;
    uint64_t res = (uint64_t)arr[0] << 56 |
                   (uint64_t)arr[1] << 48 |
                   (uint64_t)arr[2] << 40 |
                   (uint64_t)arr[3] << 32 |
                   (uint64_t)arr[4] << 24 |
                   (uint64_t)arr[5] << 16 |
                   (uint64_t)arr[6] << 8 |
                   (uint64_t)arr[7];
    memcpy(dest, &res, 8);
}
//-----------------------------------------------------------------------------
/*! Converts a 16-bit number to big-endian and writes it to a stream
 * @param number the number to be converted and written
 * @param stream the destination stream
 */
void writeBigEndian16(uint16_t number, std::ostream& stream)
{
    char buffer[2];
    toBigEndian16(number, buffer);
    stream.write(buffer, 2);
}
//-----------------------------------------------------------------------------
/*! Converts a 32-bit number to big-endian and writes it to a stream
 * @param number the number to be converted and written
 * @param stream the destination stream
 */
void writeBigEndian32(uint32_t number, std::ostream& stream)
{
    char buffer[4];
    toBigEndian32(number, buffer);
    stream.write(buffer, 4);
}
//-----------------------------------------------------------------------------
/*! Converts a 64-bit number to big-endian and writes it to a stream
 * @param number the number to be converted and written
 * @param stream the destination stream
 */
void writeBigEndian64(uint64_t number, std::ostream& stream)
{
    char buffer[8];
    toBigEndian64(number, buffer);
    stream.write(buffer, 8);
}
//-----------------------------------------------------------------------------
} // namespace ByteOrder
  //-----------------------------------------------------------------------------