#include <Sphere.h>

void Sphere::build()
{
    // assert(_stacks > 3 && _slices > 3);

    // create vertex array
    unsigned int numV = (_stacks + 1) * (_slices + 1);
    // std::vector<Vertex> vertices(numV);
    // vertices.resize(numV);

    P.resize(numV);
    N.resize(numV);
    Tc.resize(numV);
    C.resize(numV);
    I32.resize(numV);

    float  theta, dtheta; // angles around x-axis
    float  phi, dphi;     // angles around z-axis
    SLuint i, j;          // loop counters
    SLuint iv = 0;

    // init start values
    theta  = 0.0f;
    dtheta = Utils::PI / _stacks;
    dphi   = 2.0f * Utils::PI / _slices;

    // Define vertex position & normals by looping through all _stacks
    for (i = 0; i <= _stacks; ++i)
    {
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        phi             = 0.0f;

        // Loop through all _slices
        for (j = 0; j <= _slices; ++j)
        {
            if (j == _slices) phi = 0.0f;

            // define first the normal with length 1
            N[iv].x = sin_theta * cos(phi);
            N[iv].y = sin_theta * sin(phi);
            N[iv].z = cos_theta;

            // set the vertex position w. the scaled normal
            P[iv].x = _radius * N[iv].x;
            P[iv].y = _radius * N[iv].y;
            P[iv].z = _radius * N[iv].z;

            // set the texture coords.
            Tc[iv].x = asin(N[iv].x) / Utils::PI + 0.5f;
            Tc[iv].y = asin(N[iv].y) / Utils::PI + 0.5f;

            C[iv].x = 255;
            C[iv].y = 0;
            C[iv].z = 0;
            C[iv].a = 255;

            phi += dphi;
            iv++;
        }
        theta += dtheta;
    }

    // create Index array x
    unsigned int numI = (SLuint)(_slices * _stacks * 2 * 3);
    I32.resize(numI);
    SLuint ii = 0, iV1, iV2;

    for (i = 0; i < _stacks; ++i)
    {
        // index of 1st & 2nd vertex of stack
        iV1 = i * (_slices + 1);
        iV2 = iV1 + _slices + 1;

        for (j = 0; j < _slices; ++j)
        { // 1st triangle ccw
            I32[ii++] = iV1 + j;
            I32[ii++] = iV2 + j;
            I32[ii++] = iV2 + j + 1;
            // 2nd triangle ccw
            I32[ii++] = iV1 + j;
            I32[ii++] = iV2 + j + 1;
            I32[ii++] = iV1 + j + 1;
        }
    }
}
