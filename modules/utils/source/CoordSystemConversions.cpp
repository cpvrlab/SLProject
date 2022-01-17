#include <CoordSystemConversions.h>

namespace CoordSystemConversions
{

void cv2gl4x4f(float* src, float* dest)
{
    convert4x4f<POS_X, NEG_Y, NEG_Z, false, false>(src, dest);
}

} // namespace CoordSystemConversions