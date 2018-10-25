//#############################################################################
//  File:      SLVec2f.java
//  Purpose:   2 Component vector class
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;

public class SLVec2f 
{
	/**
	 * Predefined Vectors
	 */
	public static final SLVec2f ZERO = new SLVec2f(0f, 0f);
	
	/**
	 * Coordinates
	 */
	public float x;
	public float y;
	
	//Constructors
	
	public SLVec2f(float x, float y) 
    {
		this.x = x;
		this.y = y;
	}
	
	public SLVec2f(SLVec2f vector) 
    {
		this.x = vector.x;
		this.y = vector.y;
	}
	
	public SLVec2f() 
    {
		this(SLVec2f.ZERO);
	}
	
	public SLVec2f(float[] coordinates) 
    {
		if(coordinates==null || coordinates.length<2)
			throw new IllegalArgumentException("Coordinate Array must contain at least 3 dimensions");
		this.x = coordinates[0];
		this.y = coordinates[1];
	}
	
	public SLVec2f add(SLVec2f vec) 
    {
		this.x += vec.x;
		this.y += vec.y;
		return this;
	}

	public SLVec2f add(float s) 
    {
		this.x += s;
		this.y += s;
		return this;
	}
	
	public static SLVec2f add(SLVec2f vec1, SLVec2f vec2) 
    {
		return new SLVec2f(vec1.x + vec2.x, vec1.y + vec2.y);
	}
	
	public static SLVec2f add(SLVec2f vec1, float s) 
    {
		return new SLVec2f(vec1.x + s, vec1.y + s);
	}
	
	
	public SLVec2f subtract(SLVec2f vec) 
    {
		this.x -= vec.x;
		this.y -=vec.y;
		return this;
	}
	
	public SLVec2f subtract(float s) 
    {
		this.x -= s;
		this.y -= s;
		return this;
	}
	
	public static SLVec2f subtract(SLVec2f vec1, SLVec2f vec2) 
    {
		return new SLVec2f(vec1.x - vec2.x, vec1.y - vec2.y);
	}
	
	public static SLVec2f subtract(SLVec2f vec1, float s) 
    {
		return new SLVec2f(vec1.x - s, vec1.y - s);
	}
	

	public SLVec2f divide(SLVec2f vec) 
    {
		if(vec.x==0f || vec.y==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		this.x /= vec.x;
		this.y /=vec.y;
		return this;
	}
	
	public SLVec2f divide(float s) 
    {
		if(s==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		this.x /= s;
		this.y /= s;
		return this;
	}
	
	public static SLVec2f divide(SLVec2f vec1, SLVec2f vec2) 
    {
		if(vec2.x==0f || vec2.y==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		return new SLVec2f(vec1.x / vec2.x, vec1.y / vec2.y);
	}
	
	public static SLVec2f divide(SLVec2f vec1, float s) 
    {
		if(s==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		return new SLVec2f(vec1.x / s, vec1.y / s);
	}
	
	
	public SLVec2f multiply(SLVec2f vec) 
    {
		this.x *= vec.x;
		this.y *=vec.y;
		return this;
	}
	
	public SLVec2f multiply(float s) 
    {
		this.x *= s;
		this.y *= s;
		return this;
	}
	
	public static SLVec2f multiply(SLVec2f vec1, SLVec2f vec2) 
    {
		return new SLVec2f(vec1.x * vec2.x, vec1.y * vec2.y);
	}
	
	public static SLVec2f multiply(SLVec2f vec1, float s) 
    {
		return new SLVec2f(vec1.x * s, vec1.y * s);
	}
	
	public float dot(SLVec2f vec) 
    {
		return SLVec2f.dot(this, vec);
	}
	
	public static float dot(SLVec2f u, SLVec2f v) 
    {
		return (u.x * v.x) + (u.y * v.y);
	}
	
	public SLVec2f negate() 
    {
		return this.multiply(-1f);
	}
	
	public static SLVec2f negate(SLVec2f vec) 
    {
		return vec.multiply(-1f);
	}
	
	public boolean isApproxEqual(SLVec2f vec, float tolerance) 
    {
		return Math.abs(this.x-vec.x) <= tolerance && Math.abs(this.y-vec.y) <= tolerance;
	}
	
	public float length() 
    {
	      return (float)Math.sqrt(this.x*this.x + this.y*this.y);
	}
	public float length2() 
    {
	      return (this.x*this.x + this.y*this.y );
	}
	
	public SLVec2f normalize() 
    {
		return this.divide(length());
	}

	public float[] toArray() 
    {
		return new float[] {x, y};
	}
	
	@Override
	public boolean equals(Object o) 
    {
		if(o instanceof SLVec2f)
			return isApproxEqual((SLVec2f)o, 0.0f);
		return false;
	}
	
	@Override
	public Object clone() 
    {
		return new SLVec2f(this);
	}
	
	public static float[] toArray(SLVec2f[] vecs) 
    {
		float[] arr = new float[vecs.length*3];
		for(int i=0;i<vecs.length;i++) 
        {
			SLVec2f v = vecs[i];
			arr[i*2+0] = v.x;
			arr[i*2+1] = v.y;
		}
		return arr;	
	}
}
