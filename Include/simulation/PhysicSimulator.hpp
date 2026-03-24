#pragma once 
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <torch/script.h>
#include <iostream>
using namespace std;

namespace Physics{

    class PhysicString{

        private:


            int n_;
            float k_, c_, m_, L0_, friction_;

            Eigen::VectorXd state_;   //4n vector such as [xi,yi,vxi,vyi]. vxi and vyi could eventually be replaced by old pos depending of the verlet usage
            
            //We define those variables in order to logg them as data for the GNN future training

            // -------NODES VARIABLES----------
            std::vector<Eigen::Vector2d> velocity_;
            std::vector<Eigen::Vector2d> acceleration_;
            std::vector<bool> is_fixed_;

            // -------EDGES VARIABLES----------
            std::vector<float>           L_;    
            std::vector<float>           tension_;   
            std::vector<Eigen::Vector2d> direction_;

            

            //some inline function that will help access index from the state vector
            inline int indexX(int i) const { return i;}
            inline int indexY(int i) const { return n_ + i;}
            inline int indexVX(int i) const { return 2 * n_ + i;}
            inline int indexVY(int i) const { return 3 * n_ + i;}

            //get the data from the state vector
            inline float getX(int i) const { return state_(indexX(i));}
            inline float getY(int i) const { return state_(indexY(i));}
            inline float getVX(int i) const { return state_(indexVX(i));}
            inline float getVY(int i) const { return state_(indexVY(i));}

            //set the data in the state vector
            inline void setX(int i, float val){ state_(indexX(i)) = val;}
            inline void setY(int i, float val){ state_(indexY(i)) = val;}
            inline void setVX(int i, float val){ state_(indexVX(i)) = val;}
            inline void setVY(int i, float val){ state_(indexVY(i)) = val;}

            //get the position vector from the state vector
            Eigen::Vector2d getPos(int i)  {return Eigen::Vector2d(getX(i),getY(i));}
            Eigen::Vector2d getVel(int i)  {return Eigen::Vector2d(getVX(i),getVY(i));}

            //set the state vector with a vector
            void setPos(int i, const Eigen::Vector2d vec)  {setX(i, vec.x()); setY(i,vec.y());}
            void setVel(int i, const Eigen::Vector2d vec)  {setVX(i, vec.x()); setVY(i,vec.y());}


            //force compute functions
            Eigen::Vector2d springForce(int i, int j);
            Eigen::Vector2d dampingForce(int i);




            // ------------ GNN VARIABLES ------------
            torch::jit::script::Module model_;

            // NORMALISATION VARIABLES

            torch::Tensor edge_index_;  
            torch::Tensor x_;           
            torch::Tensor edge_attr_;
            torch::Tensor acc_;
            std::vector<float> old_px_, old_py_; //temporary variable that stores the old pos for verlet integretion


            //WARNING !!!!! THOSE VARIABLES WERE SIMPLY COPY PASTED FROM THE CONSTRUCTOR OF THE Dataset.py OF THE MODEL 

            const float MEAN_X = 599.687266f;
            const float STD_X  = 44.812275f;
            const float MEAN_Y = 86.578203f;
            const float STD_Y  = 102.863401f;
            const float MEAN_VX = 0.162586f;
            const float STD_VX  = 571.353268f;
            const float MEAN_VY = -11.805742f;
            const float STD_VY  = 873.642648f;
            const float MEAN_AX = 10.174973f;
            const float STD_AX  = 15685.608690f;
            const float MEAN_AY = -12.017166f;
            const float STD_AY  = 23240.157774f;
            const float MEAN_LENGTH = 78.539048f;
            const float STD_LENGTH  = 68.355865f;
            const float MEAN_TENSION = -8561.714673f;
            const float STD_TENSION  = 20506.761477f;
            const float MEAN_DIR_X = 0.000376f;
            const float STD_DIR_X  = 0.475979f;
            const float MEAN_DIR_Y = 0.633448f;
            const float STD_DIR_Y  = 0.610113f;

        public:


            PhysicString(int n, float k, float c, float m, float L0, float friction);

            void initializeLine(const Eigen::Vector2d& start, const Eigen::Vector2d& end);

            void stepEuler(float dt);               //update the state with Euler method

            void stepVerlet(float dt);              //update the state with verlet intégration method

            void stepGNN(float dt);                 //update the state with GNN physic and verlet

            const Eigen::VectorXd& getState() const { return state_; } //getter
            void setState(const Eigen::VectorXd& s) { state_ = s; }    //setter

            int getNumPoints() const { return n_; }
            int getStateDim() const { return 4 * n_; }  
            
            Eigen::Vector2d getPointPosition(int i) const { 
                return Eigen::Vector2d(getX(i), getY(i)); 
            }

            Eigen::Vector2d getOldPosition(int i) const { // get the second point of the state inside the state vector
                return Eigen::Vector2d(getVX(i),getVY(i)); 
            }

            Eigen::Vector2d getPointVelocity(int i) const { 
                return velocity_[i]; 
            }

            Eigen::Vector2d getPointAcceleration(int i) const { 
                return acceleration_[i]; 
            }

            bool isFixed(int i) const { 
                return is_fixed_[i]; 
            }

            float getLength(int i) const { 
                return L_[i]; 
            }

            float getTension(int i) const { 
                return tension_[i]; 
            }

            Eigen::Vector2d getDirection(int i) const { 
                return direction_[i]; 
            }

            void setPointPosition(int i, const Eigen::Vector2d& pos) {
                setX(i, pos.x());
                setY(i, pos.y());
            }
            
            void setOldPosition(int i, const Eigen::Vector2d& vel) {
                setVX(i, vel.x());
                setVY(i, vel.y());
            }

            void setFixed(int i, bool v) { 
                is_fixed_[i] = v; 
            }
    };
}