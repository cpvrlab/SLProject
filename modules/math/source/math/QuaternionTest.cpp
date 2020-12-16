#include <SL.h>
#include <SLVec3.h>
#include <SLMat3.h>
#include <SLMat4.h>
#include <SLQuat4.h>

int main()
{  
   float angleDEG;
   SLVec3f axis;
   SLQuat4f q1;

   cout << "Quaternion q1: 45deg. around y-axis: ";
   SLMat3f m1(45, 0,1,0);
   q1.fromMat3(m1);
   q1.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl << endl;

   cout << "Quaternion q1: 45deg. around y-axis: ";
   q1.fromAngleAxis(45, SLVec3f(0,1,0));
   q1.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl << endl;
   
   cout << "Quaternion to matrix: 45deg. around y-axis: ";
   SLMat3f m2 = q1.toMat3();
   m2.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl << endl;

   cout << "Quaternion q2: 45 deg. around y-axis" << endl;
   cout << "Quaternion q3 = q1 x q2: 90 deg. around y-axis: ";
   SLQuat4f q2(45, SLVec3f(0,1,0));
   SLQuat4f q3 = q1.rotated(q2);
   q3.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl << endl;

   float t = 0.1f;
   for (int i=0; i<9; ++i)
   {
      cout << "Quaternion q1.slerp(" << t << ", q3): ";
      q1.slerp(q3, t).toAngleAxis(angleDEG, axis);
      cout << angleDEG << ", " << axis << endl;

      cout << "Quaternion q1.lerp (" << t << ", q3): ";
      q1.lerp(q3, t).toAngleAxis(angleDEG, axis);
      cout << angleDEG << ", " << axis << endl;
      t += 0.1f;
   }
   cout << endl;
  
   cout << "Quaternion q1: 45deg. around y-axis: " << endl;
   cout << "Quaternion q2 = q1.inverted(): ";
   q1.fromAngleAxis(45, SLVec3f(0,1,0));
   q2 = q1.inverted();
   q2.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl;
   cout << "Quaternion q3 = q1 * q2: ";
   q3 = q1 * q2;
   q3.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl << endl;

   cout << "Quaternion q1.fromEuler(0,45,0): " << endl;
   q1.fromEuler(0, 45, 0);
   q1.toAngleAxis(angleDEG, axis);
   cout << angleDEG << ", " << axis << endl;

	return 0;
}

