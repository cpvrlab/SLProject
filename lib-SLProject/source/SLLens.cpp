//#############################################################################
//  File:      SLLens.cpp
//  Author:    Philipp Jüni
//  Date:      October 2014
//  Copyright: Marcus Hudritsch, Philipp Jüni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

using namespace std;
#include <SLMaterial.h>
#include <SLLens.h>

//-----------------------------------------------------------------------------
/*!
SLLens::SLLens ctor for lens revolution object around the y-axis. <br>
Create the lens with the eye prescription card.
\image html EyePrescriptionCard.png
The first values in the prescription card is called Sphere. It is also the
diopter of the front side of the lens. <br>
The second value from the prescription card is called Cylinder. The sum from 
the spere and the cylinder is the diopter of the back side of the lens. <br>
The diopter is the inverse of the focal distance (f) of the lens. <br>
To correct myopic, negative diopters are used. <br>
To correct presbyopic, positive diopters are used.<br>
\image html Lens.png

\param sphere SLfloat taken from the eyeglass passport.
\param cylinder SLfloat taken from the eyeglass passport.
\param diameter SLfloat The diameter (h) of the lens
\param thickness SLfloat The space between the primary planes of lens sides (d)
\param stacks SLint
\param slices SLint
\param name SLstring of the SLRevolver Mesh
\param mat SLMaterial* The Material of the lens

The diopter of the front side is equal to the sphere. <br>
The diopter of the backside is the sum of the spehere and the cylinder. <br>
From the diopter, the radius (R1, R2) can be calculated: <br>
radiusFront = (LensMaterial - OutsideMaterial) / FrontDiopter) * diameter;<br>
radiusBack = (OutsideMaterial - LensMaterial) / BackDiopter) * diameter;<br>
*/
SLLens::SLLens(double sphere, 
               double cylinder, 
               SLfloat diameter,
               SLfloat thickness,
               SLint stacks, 
               SLint slices, 
               SLstring name, 
               SLMaterial* mat) : SLRevolver(name)
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices >  0 && "Error: Not enough stacks.");

    SLfloat diopterBot = (SLfloat)sphere; // D1 = sphere
    SLfloat diopterTop = (SLfloat)(sphere + cylinder); // D2 = sphere + cylinder

    init(diopterBot, diopterTop, diameter, thickness, stacks, slices, mat);
}

/*!
SLLens::SLLens ctor for lens revolution object around the y-axis. <br>
Create the lens with radius.<br>
To correct presbyopic (far-sightedness) a converging lens is needed.
To correct myopic (short-sightedness) a diverging lens is needed.
\image html Lens.png

\param radiusBot SLfloat The radius of the front side of the lens
\param radiusTop SLfloat The radius of the back side of the lens
\param diameter SLfloat The diameter (h) of the lens
\param thickness SLfloat The space between the primary planes of lens sides (d)
\param stacks SLint
\param slices SLint
\param name SLstring of the SLRevolver Mesh
\param mat SLMaterial* The Material of the lens

Positive radius creates a convex lens side. <br>
Negative radius creates a concave lens side. <br>
Setting the radius to 0 creates a plane. <br>
Combine the two radius to get the required lens.
*/
SLLens::SLLens(SLfloat radiusBot,
               SLfloat radiusTop,
               SLfloat diameter,
               SLfloat thickness,
               SLint stacks,
               SLint slices,
               SLstring name,
               SLMaterial* mat) : SLRevolver(name)
{
    SLfloat nOut = 1.00;            // kn material outside lens
    SLfloat nLens = mat->kn();      // kn material of the lens
    SLfloat diopterBot = (SLfloat)((nLens - nOut) * diameter / radiusBot);
    SLfloat diopterTop = (SLfloat)((nOut - nLens) * diameter / radiusTop);

    init(diopterBot, diopterTop, diameter, thickness, stacks, slices, mat);
}

/*! 
\brief Initialize the lens
\param diopterBot SLfloat The diopter of the bot (front) part of the lens
\param diopterTop SLfloat The diopter of the top (back) part of the lens
\param diameter SLfloat The diameter of the lens
\param thickness SLfloat d The space between the primary planes of lens sides
\param stacks SLint
\param slices SLint
\param mat SLMaterial* The Material of the lens
*/
void SLLens::init(SLfloat diopterBot, 
                  SLfloat diopterTop, 
                  SLfloat diameter, 
                  SLfloat thickness, 
                  SLint stacks, 
                  SLint slices, 
                  SLMaterial* mat)
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices >  0 && "Error: Not enough stacks.");

    _diameter = diameter;
    _thickness = thickness;
    _stacks = stacks;
    _slices = slices;
    _pointOutput = false; // cout the coordinates of each point of the lens

    // Gullstrand-Formel
    // D = D1 + D2 - delta * D1 * D2
    SLfloat diopter = diopterBot + diopterTop;

    // calc radius
    SLfloat nOut = 1.00;         // kn material outside lens
    SLfloat nLens = mat->kn();   // kn material of the lens
    SLfloat delta = _thickness / nLens; // d/n

    // calc radius
    _radiusBot = (SLfloat) ((nLens - nOut) / diopterBot) * _diameter;
    _radiusTop = (SLfloat) ((nOut - nLens) / diopterTop) * _diameter;

    if (_pointOutput)
        cout << " radiusBot: " << _radiusBot <<
                " radiusTop: " << _radiusTop << endl;

    // generate lens
    generateLens(_radiusBot, _radiusTop, mat);
}

/*! 
\brief Generate the lens with given front and back radius
\param radiusBot radius of the lens front side
\param radiusTop radius of the lens back side
\param mat the material pointer that is passed to SLRevolver
*/
void SLLens::generateLens(SLfloat radiusBot, SLfloat radiusTop, SLMaterial* mat)
{
    _smoothFirst = true;
    _smoothLast = true;
    _revAxis.set(0, 1, 0);

    if (_diameter > 0)
    {
        SLfloat x = generateLensBot(radiusBot);
        x = generateLensTop(radiusTop);
        if (x == 0)
            buildMesh(mat);
        else
            cout << "error in lens calculation: (x = " << x << ")" << endl;
    }
    else
        cout << "invalid lens diameter: " << _diameter << endl;
}

/*! 
\brief Generate the bottom (front) part of the lens
\param radius of the lens
\return x the x coordinate of the last point of the bulge
*/
SLfloat SLLens::generateLensBot(SLfloat radius)
{
    // Point
    SLVec3f p;
    SLfloat y;
    SLfloat x = 0;

    SLfloat sagitta = calcSagitta(radius);
    SLint halfStacks = _stacks / 2;

    // check if lens has a deep
    if ((sagitta >= 0))
    {
        SLfloat alphaAsin = _diameter / (2.0f * radius);
        alphaAsin = (alphaAsin > 1) ? 1 : alphaAsin;  // correct rounding errors
        alphaAsin = (alphaAsin < -1) ? -1 : alphaAsin;// correct rounding errors
        SLfloat alphaRAD = 2.0f * (SLfloat)asin(alphaAsin);
        SLfloat alphaDEG = alphaRAD * SL_RAD2DEG;
        SLfloat dAlphaRAD = (alphaRAD * 0.5f) / halfStacks;
        SLfloat dAlphaDEG = dAlphaRAD * SL_RAD2DEG;

        SLfloat yStart1 = -sagitta;
        SLfloat currentAlphaRAD = -SL_HALFPI;
        SLfloat currentAlphaDEG = currentAlphaRAD * SL_RAD2DEG;
        SLfloat radiusAmount1 = radius;
        SLfloat yTranslate1 = radius - sagitta;

        // settings for negative radius
        if (radius < 0)
        {
            yStart1 = 0;
            currentAlphaRAD = SL_HALFPI;
            radiusAmount1 = -radius;
            yTranslate1 = radius;
        }

        y = yStart1;
        // set start point
        p.x = (SLfloat) x;
        p.y = (SLfloat) y;
        p.z = 0;
        _revPoints.push_back(p);
        if (_pointOutput) cout << currentAlphaDEG << "  x: " << x << "  y: " << y << endl;
        
        // draw bottom part of the lens
        for (int i = 0; i < halfStacks; i++)
        {
            // change angle
            currentAlphaRAD += dAlphaRAD;
            currentAlphaDEG = currentAlphaRAD*SL_RAD2DEG;

            // calc x
            x = cos(currentAlphaRAD) * radiusAmount1;

            // calc y
            if ((i + 1 == halfStacks) && (radius >= 0))
                y = 0;
            else
                y = ((sin(currentAlphaRAD)) * radiusAmount1 + yTranslate1);

            if (_pointOutput) cout << currentAlphaDEG << "  x: " << x << "  y: " << y << endl;

            // set point
            p.x = x;
            p.y = y;
            p.z = 0;
            _revPoints.push_back(p);
        }
    }
    else
    {
        SLfloat cutX = (_diameter / 2) / halfStacks;

        // Draw plane
        for (int i = 0; i <= halfStacks-1; i++)
        {
            x = cutX * i;
            y = 0;

            // set point
            p.x = x;
            p.y = y;
            p.z = 0;
            _revPoints.push_back(p);
            if (_pointOutput) cout << "0" << "  x: " << x << "  y: " << y << " _B" << endl;
        }
    }
    return x;
}

/*! 
\brief Generate the top (back) part of the lens
\param radius of the lens
\return x the x coordinate of the last point of the bulge
*/
SLfloat SLLens::generateLensTop(SLfloat radius)
{
    // Point
    SLVec3f p;
    SLfloat x = _diameter / 2;
    SLfloat y;// = yStart2;

    SLfloat sagitta = calcSagitta(radius);
    SLint halfStacks = _stacks / 2;

    // check if the lens has a deep
    if ((sagitta >= 0))
    {
        SLfloat yStart2 = _thickness;
        SLfloat radiusAmount2 = radius;
        SLfloat yTranslate2 = -radius + sagitta;

        SLfloat betaAsin = _diameter / (2.0f * radius);
        betaAsin = (betaAsin > 1) ? 1 : betaAsin;  // correct rounding errors
        betaAsin = (betaAsin < -1) ? -1 : betaAsin;// correct rounding errors
        SLfloat betaRAD = 2.0f * (SLfloat)asin( betaAsin );
        SLfloat betaDEG = betaRAD * SL_RAD2DEG;
        SLfloat currentBetaRAD = SL_HALFPI - (betaRAD * 0.5f);

        // settings for negative radius
        if (radius < 0)
        {
            currentBetaRAD = -SL_HALFPI - (betaRAD * 0.5f);
            yStart2 = sagitta + _thickness;
            radiusAmount2 = -radius;
            yTranslate2 = -radius;
        }

        SLfloat currentBetaDEG = currentBetaRAD * SL_RAD2DEG;
        SLfloat dBetaRAD = (betaRAD  * 0.5f) / halfStacks;
        SLfloat dBetaDEG = dBetaRAD * SL_RAD2DEG;

        // set start point
        y = yStart2;
        p.x = (SLfloat) x;
        p.y = (SLfloat) y;
        p.z = 0;
        _revPoints.push_back(p);
        if (_pointOutput) cout << currentBetaDEG << "  x: " << x << "  y: " << y << endl;

        // draw top part of the lens
        for (int i = 0; i < halfStacks; i++)
        {
            // change angle
            currentBetaRAD += dBetaRAD;
            currentBetaDEG = currentBetaRAD*SL_RAD2DEG;

            // calc y
            y = (((sin(currentBetaRAD)) * (radiusAmount2)) + yTranslate2 + _thickness);

            // calc x
            if ((i + 1 == halfStacks))
            {
                x = 0;
                if (radius < 0)
                    y = _thickness;
            }
            else
                x = cos(currentBetaRAD) * radiusAmount2;

            if (_pointOutput) cout << currentBetaDEG << "  x: " << x << "  y: " << y << endl;

            // set point
            p.x = x;
            p.y = y;
            p.z = 0;
            _revPoints.push_back(p);
        }
    }
    else
    {
        SLfloat cutX = x / halfStacks;

        // Draw plane
        for (int i = halfStacks-1; i >= 0; i--)
        {
            x = cutX * i;
            y = _thickness;

            // set point
            p.x = x;
            p.y = y;
            p.z = 0;
            _revPoints.push_back(p);
            if (_pointOutput) cout << "0" << "  x: " << x << "  y: " << y << " _T" << endl;
        }
    }
    return x;
}

/*! 
\brief Calculate the sagitta (s) for a given radius (r) and diameter (l+l) 
l: half of the lens diameter
\param radius r of the lens
\return sagitta s of the lens
\image html Sagitta.png
http://en.wikipedia.org/wiki/Sagitta_%28geometry%29
*/
SLfloat SLLens::calcSagitta(SLfloat radius)
{
    // take the amount of the radius
    SLfloat radiusAmount = (radius < 0) ? -radius : radius;
    SLfloat l = _diameter * 0.5f;

    // sagitta = radius - sqrt( radius*radius - l*l )
    // calc this part to sort out negative numbers in square root
    SLfloat part = radiusAmount*radiusAmount - l*l;

    // set sagitta negative if no bulge is given -> plane
    SLfloat sagitta = (part >= 0) ? (radiusAmount - sqrt(part)) : -1;
    return sagitta;
}
//-----------------------------------------------------------------------------
