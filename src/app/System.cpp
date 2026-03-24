#include "../../include/app/System.hpp"
#include <Eigen/Dense>

namespace App{
    System::System(int n,
           float k,
           float c,
           float m,
           float L0,
           float dt,
           float friction,
           float kappa) : 
           n_(n),
           dt_(dt),
           renderer_(1200,1200),
           myString_(n_, k, c, m, L0, friction),
           meas_(n_, 4*n_),
           ukf_(myString_.getState(), myString_, meas_, n_, 4*n_, 4*(n-1), kappa),
           frame_id_(0)
           {
                Eigen::Vector2d anchor(600, 0);
                Eigen::Vector2d tip(600, 400);
                myString_.initializeLine(anchor, tip);
                ukf_.setState(myString_.getState());
           }


    void System::run(){
        while (renderer_.isOpen()) {
            renderer_.handleEvents(myString_);

            myString_.stepVerlet(dt_);
            Eigen::VectorXd realState = myString_.getState();
            


            ukf_.predict(dt_);
            ukf_.update(realState);
            Eigen::VectorXd ukfState  = ukf_.getState();

            renderer_.clear();
            renderer_.drawState(realState, n_, sf::Color::White); 
            renderer_.drawState(ukfState,  n_, sf::Color::Green); 
            renderer_.display();
        }
    }

    void System::runDataCollection() {
        
        Physics::Logger logger(std::string(DATA_DIR) + "/nodes.csv", std::string(DATA_DIR) + "/edges.csv");


        const int num_snapshots   = 50;
        const int log_every       = 5;
        const int frames_per_snap = 300;

        std::srand(42);

        // Laisse la corde se stabiliser avant de commencer
        for (int warmup = 0; warmup < 200; warmup++)
            myString_.stepVerlet(dt_);

        for (int snap = 0; snap < num_snapshots; snap++) {

            // Simule un clic sur une position aléatoire
            float rx = static_cast<float>(std::rand() % 1200);
            float ry = static_cast<float>(std::rand() % 1200);
            Eigen::Vector2d target(rx, ry);

            int tipIndex = myString_.getNumPoints() - 1;
            myString_.setPointPosition(tipIndex, target);
            myString_.setOldPosition(tipIndex, target);
            myString_.setFixed(tipIndex, true);

            // Simule le drag maintenu pendant quelques frames puis relâché
            for (int frame = 0; frame < frames_per_snap; frame++) {
                
                // Relâche le tip après 30 frames pour laisser la corde osciller
                if (frame == 30)
                    myString_.setFixed(tipIndex, false);

                myString_.stepVerlet(dt_);
                ukf_.predict(dt_);
                ukf_.update(myString_.getState());

                if (frame % log_every == 0)
                    logger.log(myString_, snap, frame);
            }
        }
    }


    void System::runGNN() {

        const int log_every  = 5;
        int frame = 0;
        
        while (renderer_.isOpen()) {
            

            if (frame % log_every == 0)
            renderer_.handleEvents(myString_);

            myString_.stepGNN(dt_);
            Eigen::VectorXd GNNState = myString_.getState();


            renderer_.clear();
            renderer_.drawState(GNNState, n_, sf::Color::White); 
            renderer_.display();

            frame ++;

        }
        
    }
}