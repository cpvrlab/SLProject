void doStereoSeparation()
{
    // See SLProjType in SLEnum.h
    if (u_camProjType > 8) // stereoColors
    {
        // Apply color filter but keep alpha
        o_fragColor.rgb = u_camStereoColors * o_fragColor.rgb;
    }
    else if (u_camProjType == 6) // stereoLineByLine
    {
        if (mod(floor(gl_FragCoord.y), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
            discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
            discard;
        }
    }
    else if (u_camProjType == 7) // stereoColByCol
    {
        if (mod(floor(gl_FragCoord.x), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
            discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
            discard;
        }
    }
    else if (u_camProjType == 8) // stereoCheckerBoard
    {
        bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
        bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
        if (h==v)// both even or odd
        {
            if (u_camStereoEye ==-1)
            discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
            discard;
        }
    }
}
