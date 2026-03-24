#include "../../include/simulation/PhysicSimulator.hpp"

namespace Physics {

    PhysicString::PhysicString(int n, float k, float c, float m, float L0, float friction) : n_(n) , k_(k), c_(c), m_(m), L0_(L0), friction_(friction)
                {
                    model_ = torch::jit::load(MODEL_PATH, torch::kCPU);
                    model_.eval();
                    state_.resize(4 * n);
                    state_.setZero();

                    velocity_.resize(n_, Eigen::Vector2d::Zero());
                    acceleration_.resize(n_, Eigen::Vector2d::Zero());
                    is_fixed_.resize(n_, false);
                    is_fixed_[0] = true;// anchor always fixed

                    L_.resize(n_ - 1, 0.0f);
                    tension_.resize(n_ - 1, 0.0f);
                    direction_.resize(n_ - 1, Eigen::Vector2d::Zero());

                    //GNN variables

                    old_px_.resize(n_, 0.0f);
                    old_py_.resize(n_, 0.0f);
                    
                    edge_index_ = torch::zeros({2, 2*(n_-1)}, torch::kLong);
                    for (int i = 0; i < n_-1; i++)
                    {
                        edge_index_[0][i]           = i;      // src → dst
                        edge_index_[1][i]           = i + 1;
                        edge_index_[0][i + (n_-1)] = i + 1;  // dst → src
                        edge_index_[1][i + (n_-1)] = i;
                    }

                    // x : [n, 5] — (x, y, vx, vy, is_fixed)
                    x_ = torch::zeros({n_, 5}, torch::kFloat);

                    // edge_attr : [2*(n-1), 4] — (length, tension, dir_x, dir_y) both sides
                    edge_attr_ = torch::zeros({2*(n_-1), 4}, torch::kFloat);

                    acc_ = torch::zeros({n_, 2}, torch::kFloat);
                }


    //Creating a function that is able to give a straight line position from the begenning with every points 
    void PhysicString::initializeLine(const Eigen::Vector2d& start, const Eigen::Vector2d& end)
        {
           Eigen::Vector2d delta = (end - start) / float(n_ - 1);    //using the first and the last point to divide the string in equal little parts

            for (int i = 0; i < n_; ++i) {
                Eigen::Vector2d p = start + delta * double(i);
                setPos(i, p);
                setVel(i, p);       
                old_px_[i] = p.x(); 
                old_py_[i] = p.y();
                
            }
        }


    Eigen::Vector2d PhysicString::springForce(int i, int j) {
        int edge_idx = std::min(i, j);  

        Eigen::Vector2d springF = getPos(i) - getPos(j);
        L_[edge_idx]         = springF.norm();
        direction_[edge_idx] = springF / L_[edge_idx];
        float ext            = L_[edge_idx] - L0_;
        tension_[edge_idx]   = ext * -k_;
        springF              = (direction_[edge_idx] * tension_[edge_idx]);
        return springF;
    }


    Eigen::Vector2d PhysicString::dampingForce(int i){  //compute the damping force
        return  getVel(i) * (-c_);
    }

    

    void PhysicString::stepEuler(float dt){  // classic integration Euler method

        std::vector<Eigen::Vector2d> acc(n_);  //creation of an acc vector for the integration
        
        setPos(0,Eigen::Vector2d(600,0)); //we fix the anchor
        setVel(0,Eigen::Vector2d(600,0));


        for ( int i = 1 ; i < n_; i++){            //Newton's 3rd law loop where we sum all the force 
            Eigen::Vector2d F(0,0);

            if (i > 0) {
                F += springForce(i, i-1);
            }
            if (i < n_ - 1) {
                F += springForce(i, i+1);
            }
            F += dampingForce(i);
            F += Eigen::Vector2d (0, 9.81* m_);

            acc[i] = F/m_;
        }

        for ( int i = 1 ; i < n_; i++){         //update state loop
            Eigen::Vector2d v = getVel(i);
            Eigen::Vector2d p = getPos(i);
            
            v += acc[i] * dt;
            p += v * dt;
            
            setVel(i, v);
            setPos(i, p);
        }
    }


    void PhysicString::stepVerlet(float dt)
    {
        // we code the verlet integration method for more efficiency
        std::vector<Eigen::Vector2d> acc(n_); // creation of an acc vector for the integration

        setPos(0, Eigen::Vector2d(600, 0));   // we fix the anchor
        setVel(0, Eigen::Vector2d(600, 0));   // we fix the first old position as the actual position

        for (int i = 1; i < n_; i++)
        {
            // Newton's 3rd law loop where we sum all the force
            Eigen::Vector2d F(0, 0);

            if (i > 0)
            {
                F += springForce(i, i - 1);
            }
            if (i < n_ - 1)
            {
                F += springForce(i, i + 1);
            }
            F += Eigen::Vector2d(0, 9.81 * 100 * m_);

            acc[i] = F / m_;
            
            acceleration_[i]       = acc[i];
        }

        for (int i = 1; i < n_; i++)
        {
            // update state loop
            Eigen::Vector2d p = getPos(i);
            Eigen::Vector2d old_p = getVel(i);

            Eigen::Vector2d v = (p - old_p) / dt;
            velocity_[i]       = v;

            setVel(i, p);  // We no longer store the velocity with the verlet update but the old position

            Eigen::Vector2d new_p = p + (p - old_p) * friction_ + acc[i] * dt * dt;
            // in the verlet integration we use the friction coefficient rather than the damping force

            setPos(i, new_p);
        }
    }

    void PhysicString::stepGNN(float dt)
    {
        setPos(0, Eigen::Vector2d(600, 0));   // we fix the anchor
        setVel(0, Eigen::Vector2d(600, 0));   // we fix the first old position as the actual position

        for (int i = 1; i < n_; i++) {
            float vx = (getX(i) - old_px_[i]) / dt;
            float vy = (getY(i) - old_py_[i]) / dt;
            velocity_[i] = Eigen::Vector2d(vx, vy);
        }

        for ( int i = 0 ; i < n_-1 ; i++){          

            Eigen::Vector2d temp = getPos(i) - getPos(i+1);
            float length = temp.norm();
            edge_attr_[i][0] = (length - MEAN_LENGTH)/STD_LENGTH;
            edge_attr_[i][1] = (((length - L0_) * -k_)- MEAN_TENSION)/STD_TENSION;
            edge_attr_[i][2] = ((temp.x() / length)- MEAN_DIR_X)/STD_DIR_X;
            edge_attr_[i][3] = ((temp.y() / length)- MEAN_DIR_Y)/STD_DIR_Y;

            edge_attr_[(n_-1) + i][0] = (length - MEAN_LENGTH)/STD_LENGTH;
            edge_attr_[(n_-1) + i][1] = (((length - L0_) * -k_)- MEAN_TENSION)/STD_TENSION;
            edge_attr_[(n_-1) + i][2] = ((-temp.x() / length)- MEAN_DIR_X)/STD_DIR_X;
            edge_attr_[(n_-1) + i][3] = ((-temp.y() / length)- MEAN_DIR_Y)/STD_DIR_Y;
        }

        for (int i = 0; i < n_; i++) {
            x_[i][0] = (getX(i) - MEAN_X) / STD_X;
            x_[i][1] = (getY(i) - MEAN_Y) / STD_Y;
            x_[i][2] = (velocity_[i].x() - MEAN_VX) / STD_VX;
            x_[i][3] = (velocity_[i].y() - MEAN_VY) / STD_VY;
            x_[i][4] = is_fixed_[i] ? 1.0f : 0.0f;
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x_);
        inputs.push_back(edge_index_);
        inputs.push_back(edge_attr_);

        std::cout << "=== Step debug ===" << std::endl;
        for (int i = 0; i < n_; i++) {
        std::cout << "node " << i << " x_norm=(" 
              << x_[i][0].item<float>() << ", " << x_[i][1].item<float>() 
              << ") v_norm=(" 
              << x_[i][2].item<float>() << ", " << x_[i][3].item<float>() << ")" << std::endl;
        }

        // model calling
        acc_ = model_.forward(inputs).toTensor();

        for (int i = 0; i < n_; i++) {
            std::cout << "acc " << i << " norm=(" 
                    << acc_[i][0].item<float>() << ", " << acc_[i][1].item<float>() << ")"
                    << " denorm=(" 
                    << acc_[i][0].item<float>() * STD_AX + MEAN_AX << ", "
                    << acc_[i][1].item<float>() * STD_AY + MEAN_AY << ")" << std::endl;
        }

        for (int i = 1; i < n_; i++) {
            float px = getX(i);
            float py = getY(i);

            float new_px = px + (px - old_px_[i]) * friction_ + (acc_[i][0].item<float>() * STD_AX + MEAN_AX) * dt * dt;
            float new_py = py + (py - old_py_[i]) * friction_ + (acc_[i][1].item<float>() * STD_AY + MEAN_AY) * dt * dt;

            old_px_[i] = px;
            old_py_[i] = py;
            setPos(i, Eigen::Vector2d(new_px, new_py));
        }

    }
}
