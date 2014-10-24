//#############################################################################
//  File:      glUtils.js
//  Purpose:   General WebGL utility functions for simple WebGL demo apps
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: M. Hudritsch, Fachhochschule Nordwestschweiz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

///////////////////////
// Utility Functions //
///////////////////////

function log(msg) 
{
    if (window.console && window.console.log) 
        window.console.log(msg);
}

///////////////////////////////
// Shader Creation Functions //
///////////////////////////////

/* 
loadShader loads the ASCII content of a shader file with an XMLHttpRequest from
the webserver and returns it as a string.
If the file can not be opened an error message is sent to stdout before the app
exits with code 1.
*/
function loadShader(url) 
{
    var req = new XMLHttpRequest();
    req.open("GET", url, false);
    req.send(null);
    if (req.status == 200) 
        return req.responseText;
    else
    {	log("File open failed: " + url);
        return null;
    }
};
    
/*
buildShader loads the shader file, creates an OpenGL shader object, compiles the 
source code and returns the handle to the internal shader object. If the 
compilation fails the compiler log is sent to the stdout before the app exits 
with code 1.
*/
function buildShader(shaderFileURL, shaderType)
{
    // Load shader file, create shader and compile it
    var shaderSource = loadShader(shaderFileURL);
    if (shaderSource==null) return;
    var shaderHandle = gl.createShader(shaderType);
    gl.shaderSource(shaderHandle, shaderSource);
    gl.compileShader(shaderHandle);

    // Check the compile status
    var compileSuccess = gl.getShaderParameter(shaderHandle, gl.COMPILE_STATUS);
    if (!compileSuccess) 
    {   var logMsg = gl.getShaderInfoLog(shaderHandle);
        log("**** Compile Error ****");
        log("In file: " + shaderFileURL);
        log("Error: " + logMsg);
        return null;
    }
    return shaderHandle;
}
    
/*
buildProgram creates a program object, attaches the shaders, links them and 
returns the OpenGL handle of the program. If the linking fails the linker log 
is sent to the stdout before the app exits with code 1.
*/
function buildProgram(shaderVertID, shaderFragID)
{
     // Create shader program, attach shaders and link them 
    var programHandle = gl.createProgram();
    gl.attachShader (programHandle, shaderVertID);
    gl.attachShader (programHandle, shaderFragID);            
    gl.linkProgram(programHandle);

    // Check the link status
    var linkSuccess = gl.getProgramParameter(programHandle, gl.LINK_STATUS);
    if (!linkSuccess && !gl.isContextLost()) 
    {   var error = gl.getProgramInfoLog (programHandle);
        log("**** Link Error ****" + error);
        return null;
    }
    return programHandle;
}
    
/*
buildVBO generates a Vertex Buffer Object (VBO) and copies the data into the
buffer on the GPU and returns the id of the buffer.
The targetTypeGL distincts between GL_ARRAY_BUFFER for attribute 
data and GL_ELEMENT_ARRAY_BUFFER for index data. The usageTypeGL distincts 
between GL_STREAM_DRAW, GL_STATIC_DRAW and GL_DYNAMIC_DRAW.
*/
function buildVBO(data, targetTypeGL, usageTypeGL)
{
    // Generate a buffer id
    var vboID = gl.createBuffer();
    
    // binds (activates) the buffer that is used next
    gl.bindBuffer(targetTypeGL, vboID);
    
    // copy data to the VBO on the GPU. The data could be delete afterwards.
    gl.bufferData(targetTypeGL, data, usageTypeGL);
    gl.bindBuffer(targetTypeGL, null);
    return vboID;
}
    
/*
buildTexture loads and build the OpenGL texture on the GPU. The image is 
loaded asynchronously with an onLoad event handler
*/
function buildTexture(imageURL, minFilter, magFilter, wrapS, wrapT) 
{
    // create texture object and image object
    var textureHandle = gl.createTexture();
    var image = new Image();
    image.src = imageURL; // needed for asynchronous loading
    
    if (typeof(minFilter)=="undefined") minFilter = gl.LINEAR_MIPMAP_LINEAR;
    if (typeof(magFilter)=="undefined") magFilter = gl.LINEAR;
    if (typeof(wrapS)=="undefined") wrapS = gl.REPEAT;
    if (typeof(wrapT)=="undefined") wrapT = gl.REPEAT;
 
    // load image with load asynchronous event handler
    image.onload = function() 
    {   
        // bind the texture as the active one
        gl.bindTexture(gl.TEXTURE_2D, textureHandle);
        
        // flip y-axis
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        
        // apply minification & magnification filter
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
        
        // apply texture wrapping modes
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapT);
        
        // copy image data to the GPU. The image can be delete afterwards
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        
        // Generate mipmaps
        if (minFilter == gl.LINEAR_MIPMAP_LINEAR)
            gl.generateMipmap(gl.TEXTURE_2D);
    }
    image.onerror = function() 
    {   log("error while loading image: '" + imageURL + "'.");
    }
 
    // return texture object (asynchronous loading, texture NOT available yet!)
    return textureHandle;
}