
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

import javax.swing.JFrame;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class HelloCube extends JPanel implements ComponentListener,
									             MouseListener, 
									             MouseMotionListener, 
									             MouseWheelListener
{
	// Private members
	private SLMat4f   m_viewMatrix;   	    // View matrix (world to camera transform)
	private SLMat4f   m_modelMatrix;   		// Model matrix (object to world transform)
	private SLMat4f   m_projectionMatrix;  	// Projection matrix (camera to normalize device coords.)
	private SLMat4f   m_viewportMatrix;    	// viewport matrix
	private SLVec3f[] m_v;                 	// array for vertices for the cube
	private float     m_camZ;              	// z-distance of camera
	private float     m_rotAngle;          	// angle of cube rotation
	   

	//We initialize the matrices and the vertices for the wire frame cube
    public HelloCube()
    {
        // Create matrices
        m_viewMatrix       = new SLMat4f();
        m_modelMatrix      = new SLMat4f();
        m_projectionMatrix = new SLMat4f();
        m_viewportMatrix   = new SLMat4f();

        // define the 8 vertices of a cube
        m_v    = new SLVec3f[8];
        m_v[0] = new SLVec3f(-0.5f,-0.5f, 0.5f); // front lower left
        m_v[1] = new SLVec3f( 0.5f,-0.5f, 0.5f); // front lower right
        m_v[2] = new SLVec3f( 0.5f, 0.5f, 0.5f); // front upper right
        m_v[3] = new SLVec3f(-0.5f, 0.5f, 0.5f); // front upper left
        m_v[4] = new SLVec3f(-0.5f,-0.5f,-0.5f); // back lower left
        m_v[5] = new SLVec3f( 0.5f,-0.5f,-0.5f); // back lower right
        m_v[6] = new SLVec3f( 0.5f, 0.5f,-0.5f); // back upper left
        m_v[7] = new SLVec3f(-0.5f, 0.5f,-0.5f); // back upper right

        m_camZ = -4;      // backwards movement of the camera
        m_rotAngle = 0;   // initial rotation angle
        
        // Create and add our keyboard listener cursor key control
        addComponentListener(this);
        addMouseListener(this);
        addMouseMotionListener(this);
        addMouseWheelListener(this);
        
        // Allow the panel to get the keyboard focus
        setFocusable(true);
        
        // Every frame the background is cleared to black
        setBackground(Color.WHITE);
        
        // Avoid flickering by double buffered painting
        setDoubleBuffered(true);
    }

    // Paints all the content of the panel every frame
    public void paint(Graphics g)
    {   
        // call the parents paint for clearing the panel
        super.paint(g);
        
        // Add rendering hints for best anti aliasing in rendering quality
        RenderingHints rh = new RenderingHints(RenderingHints.KEY_ANTIALIASING, 
        		                               RenderingHints.VALUE_ANTIALIAS_ON);
        rh.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        ((Graphics2D)g).setRenderingHints(rh);
        
     
        // view transform: move the coordinate system away from the camera
        m_viewMatrix.identity();
        m_viewMatrix.translate(0, 0, m_camZ);
     
        // model transform: rotate the coordinate system increasingly
        m_modelMatrix.identity();
        m_modelMatrix.rotate(m_rotAngle+=0.05f, 0,1,0);
        m_modelMatrix.scale(2, 2, 2);
        
        // build combined matrix out of viewport, projection, view & model matrix
        SLMat4f m = new SLMat4f();
        m.multiply(m_viewportMatrix);
        m.multiply(m_projectionMatrix);
        m.multiply(m_viewMatrix);
        m.multiply(m_modelMatrix);
        
        // transform all vertices into screen space (x & y in pixels and z as the depth) 
        SLVec3f[] v2 = new SLVec3f[8];
        for (int i=0; i < m_v.length; ++i)
        {  v2[i] = m.multiply(m_v[i]);
        }
        
        // draw front square
        g.setColor(Color.RED);
        g.drawLine((int)v2[0].x, (int)v2[0].y, (int)v2[1].x, (int)v2[1].y);
        g.drawLine((int)v2[1].x, (int)v2[1].y, (int)v2[2].x, (int)v2[2].y);
        g.drawLine((int)v2[2].x, (int)v2[2].y, (int)v2[3].x, (int)v2[3].y);
        g.drawLine((int)v2[3].x, (int)v2[3].y, (int)v2[0].x, (int)v2[0].y);
        // draw back square
        g.setColor(Color.GREEN);
        g.drawLine((int)v2[4].x, (int)v2[4].y, (int)v2[5].x, (int)v2[5].y);
        g.drawLine((int)v2[5].x, (int)v2[5].y, (int)v2[6].x, (int)v2[6].y);
        g.drawLine((int)v2[6].x, (int)v2[6].y, (int)v2[7].x, (int)v2[7].y);
        g.drawLine((int)v2[7].x, (int)v2[7].y, (int)v2[4].x, (int)v2[4].y);
        // draw from front corners to the back corners
        g.setColor(Color.BLUE);
        g.drawLine((int)v2[0].x, (int)v2[0].y, (int)v2[4].x, (int)v2[4].y);
        g.drawLine((int)v2[1].x, (int)v2[1].y, (int)v2[5].x, (int)v2[5].y);
        g.drawLine((int)v2[2].x, (int)v2[2].y, (int)v2[6].x, (int)v2[6].y);
        g.drawLine((int)v2[3].x, (int)v2[3].y, (int)v2[7].x, (int)v2[7].y);
        
        // Make sure the display is synchronized
        Toolkit.getDefaultToolkit().sync();
        
        // Dispose all graphic resources every frame
        g.dispose();
        
        repaint();
    }

    // Implement Component methods
    public void componentResized (ComponentEvent e)
    {
    	float w = this.getWidth();
    	float h = this.getHeight();
        m_projectionMatrix.perspective(50, w/h, 1.0f, 3.0f);
        m_viewportMatrix.viewport(0, 0, w, h, 0, 1);
    	System.out.println("componentResized"); 
    	repaint();
    }
    public void componentMoved   (ComponentEvent e){}
    public void componentShown   (ComponentEvent e){}
    public void componentHidden  (ComponentEvent e){}

    // Implement MouseListener methods
    public void mousePressed   (MouseEvent e)	
    {
    	// ???
    	System.out.println("mousePressed"); 
    	repaint();
    }
    public void mouseReleased  (MouseEvent e)
    {
    	// ???
    	System.out.println("mouseReleased"); 
    	repaint();
    }
    public void mouseClicked(MouseEvent e){}
    public void mouseEntered(MouseEvent e){}
    public void mouseExited(MouseEvent e){}
    
    // Implement MouseMoveListener
    public void mouseDragged(MouseEvent e)   
    {
    	// ???
    	System.out.println("mouseDragged"); 
    	repaint();
    }
    public void mouseMoved(MouseEvent e){}
    
    // Implement MouseWheelListener
    public void mouseWheelMoved(MouseWheelEvent e) 
    {	
    	// ???
    	repaint();
    }
    
    
    // The applications entry point
    public static void main(String[] args)
    {        
        JFrame myFrame = new JFrame();
        myFrame.add(new HelloCube());
        myFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        myFrame.setSize(400, 300);
        myFrame.setLocationRelativeTo(null);
        myFrame.setTitle("Hello Cube with Java");
        //myFrame.setResizable(false);
        myFrame.setVisible(true);
    }
}