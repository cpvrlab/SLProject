#ifndef WAI_MATH_H
#define WAI_MATH_H

#include <math.h>

namespace WAI
{

#define WAI_PI 3.14159265
#define WAI_PI_OVER_180 WAI_PI / 180.0f

inline float rad(float grad)
{
    float result = grad * WAI_PI_OVER_180;

    return result;
}

struct V2
{
    union
    {
        float e[2];
        struct
        {
            float x, y;
        };
        struct
        {
            float u, v;
        };
    };
};

struct V3
{
    union
    {
        float e[3];
        struct
        {
            float x, y, z;
        };
    };
};

inline V3 v3(float x, float y, float z)
{
    V3 result;

    result.x = x;
    result.y = y;
    result.z = z;

    return result;
}

inline V3 subV3(V3 v1, V3 v2)
{
    V3 result;

    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;

    return result;
}

struct M3x3
{
    float e[3][3];
};

inline M3x3 identityM3x3()
{
    M3x3 result;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (i == j)
            {
                result.e[i][j] = 1.0f;
            }
            else
            {
                result.e[i][j] = 0.0f;
            }
        }
    }

    return result;
}

inline M3x3 multM3x3(M3x3 m1, M3x3 m2)
{
    M3x3 result = {};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                float v1 = m1.e[k][j];
                float v2 = m2.e[i][k];
                result.e[i][j] += v1 * v2;
            }
        }
    }

    return result;
}

inline V3 multM3x3V3(M3x3 m, V3 v)
{
    V3 result = {};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result.e[i] += m.e[j][i] * v.e[j];
        }
    }

    return result;
}

inline M3x3 rotateXM3x3(float theta)
{
    M3x3 result = identityM3x3();

    result.e[1][1] = cos(theta);
    result.e[1][2] = sin(theta);
    result.e[2][1] = -1 * sin(theta);
    result.e[2][2] = cos(theta);

    return result;
}

struct M4x4
{
    float e[4][4];
};
}

#endif
