#include <inttypes.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    r32;
typedef double   r64;

typedef int32_t bool32;

#define kilobytes(value) ((value)*1024LL)
#define megabytes(value) (kilobytes(value) * 1024LL)
#define gigabytes(value) (megabytes(value) * 1024LL)
#define terabytes(value) (gigabytes(value) * 1024LL)

#if BUILD_DEBUG
#    define Assert(expression) \
        if (!(expression)) { *(int*)0 = 0; }
#else
#    define Assert(expression)
#endif

#define PI 3.1415926535897932384626433832795
#define DEG2RAD ((r32)(PI / 180.0f))

#define array_length(A) ((sizeof(A)) / sizeof(A[0]))

struct FrameBuffer
{
    void* memory;
    i32   width, height;
    i32   bytesPerPixel;
    i32   pitch;
};

#define ORB_PATCH_SIZE 31
#define ORB_HALF_PATCH_SIZE 15
#define ORB_EDGE_THRESHOLD 19
#define ORB_DESCRIPTOR_COUNT 32

#include "wai_orbpattern.h"

struct OrbExtractionState
{
    i32 umax[ORB_HALF_PATCH_SIZE + 1];
};

struct KeyPoint
{
    r32 x, y;
    r32 angle;
};

static inline r32 computeKeypointAngle(OrbExtractionState* state,
                                       FrameBuffer*        buffer,
                                       r32                 x,
                                       r32                 y)
{
    i32 m_01 = 0;
    i32 m_10 = 0;

    const u8* center = ((u8*)buffer->memory) + cvRound(y) * buffer->pitch + cvRound(x) * buffer->bytesPerPixel;

    // Treat the center line differently, v=0
    for (i32 u = -ORB_HALF_PATCH_SIZE; u <= ORB_HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    i32 pitch = buffer->pitch;
    for (i32 v = 1; v <= ORB_HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        i32 v_sum = 0;
        i32 d     = state->umax[v];
        for (i32 u = -d; u <= d; ++u)
        {
            i32 val_plus  = center[u + v * pitch];
            i32 val_minus = center[u - v * pitch];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    r32 result = cv::fastAtan2((float)m_01, (float)m_10);

    return result;
}

static inline u8 getOrbPatternValue(const u8*  center,
                                    const i32  bufferPitch,
                                    const r32  a,
                                    const r32  b,
                                    const i32* pattern,
                                    const i32  index)
{
    u8 result;

    result = *(center +
               cvRound(pattern[index * 2] * b + pattern[index * 2 + 1] * a) * bufferPitch +
               cvRound(pattern[index * 2] * a - pattern[index * 2 + 1] * b));

    return result;
}

static void computeOrbDescriptor(FrameBuffer*    buffer,
                                 const KeyPoint* keyPoint,
                                 const i32*      pattern,
                                 u8*             desc)
{
    Assert(buffer->bytesPerPixel == 1);

    float angle = (r32)keyPoint->angle * DEG2RAD;
    float a = (r32)cos(angle), b = (r32)sin(angle);

    const u8* center = ((u8*)buffer->memory) + cvRound(keyPoint->y) * buffer->pitch + cvRound(keyPoint->x) * buffer->bytesPerPixel;
    const i32 pitch  = buffer->pitch;

    for (int i = 0; i < ORB_DESCRIPTOR_COUNT; i++, pattern += ORB_DESCRIPTOR_COUNT)
    {
        int t0, t1, val;
        t0  = getOrbPatternValue(center, pitch, a, b, pattern, 0);
        t1  = getOrbPatternValue(center, pitch, a, b, pattern, 1);
        val = t0 < t1;
        t0  = getOrbPatternValue(center, pitch, a, b, pattern, 2);
        t1  = getOrbPatternValue(center, pitch, a, b, pattern, 3);
        val |= (t0 < t1) << 1;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 4);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 5);
        val |= (t0 < t1) << 2;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 6);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 7);
        val |= (t0 < t1) << 3;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 8);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 9);
        val |= (t0 < t1) << 4;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 10);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 11);
        val |= (t0 < t1) << 5;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 12);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 13);
        val |= (t0 < t1) << 6;
        t0 = getOrbPatternValue(center, pitch, a, b, pattern, 14);
        t1 = getOrbPatternValue(center, pitch, a, b, pattern, 15);
        val |= (t0 < t1) << 7;

        desc[i] = (u8)val;
    }
}

std::vector<KeyPoint> detectFastCorners(FrameBuffer* buffer,
                                        i32          threshold)
{
    Assert(buffer->bytesPerPixel == 1);
    std::vector<KeyPoint> result;

    cv::Mat cvImg = cv::Mat(buffer->width, buffer->height, CV_8UC1, buffer->memory, buffer->pitch);

    std::vector<cv::KeyPoint> cvKeyPoints;
    cv::FAST(cvImg, cvKeyPoints, threshold, true);

    for (cv::KeyPoint cvKeyPoint : cvKeyPoints)
    {
        KeyPoint keyPoint = {cvKeyPoint.pt.x, cvKeyPoint.pt.y, -1.0f};

        result.push_back(keyPoint);
    }

    return result;
}

std::vector<KeyPoint> detectFastCorners(OrbExtractionState* state,
                                        FrameBuffer*        buffer,
                                        i32                 threshold,
                                        i32                 borderSize)
{
    const i32 minBorderX = borderSize - 3;
    const i32 minBorderY = minBorderX;
    const i32 maxBorderX = buffer->width - borderSize + 3;
    const i32 maxBorderY = buffer->height - borderSize + 3;

    FrameBuffer fastBuffer;
    fastBuffer.width         = (maxBorderX - minBorderX);
    fastBuffer.height        = (maxBorderY - minBorderY);
    fastBuffer.bytesPerPixel = 1;
    fastBuffer.pitch         = buffer->pitch;
    fastBuffer.memory        = ((u8*)buffer->memory) +
                        minBorderY * fastBuffer.pitch +
                        minBorderX * fastBuffer.bytesPerPixel;

    std::vector<KeyPoint> result = detectFastCorners(&fastBuffer, threshold);

    for (KeyPoint& keyPoint : result)
    {
        // Bring point coordinates from fastBuffer space to
        // buffer space
        keyPoint.x += minBorderX;
        keyPoint.y += minBorderY;

        keyPoint.angle = computeKeypointAngle(state,
                                              buffer,
                                              keyPoint.x,
                                              keyPoint.y);
    }

    return result;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int descriptorDistance(const u8* a, const u8* b)
{
    const i32* pa = (i32*)a;
    const i32* pb = (i32*)b;

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v              = v - ((v >> 1) & 0x55555555);
        v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
