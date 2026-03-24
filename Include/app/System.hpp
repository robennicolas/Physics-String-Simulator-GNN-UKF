#pragma once
#include "../Simulation/PhysicSimulator.hpp"
#include "../Simulation/measurement.hpp"
#include "../Estimation/UKF.hpp"
#include "../Renderer.hpp"
#include "../Logs/Logger.hpp"
#include <Eigen/Dense>

namespace App {

class System {
public:
    System(int n,
           float k,
           float c,
           float m,
           float L0,
           float dt,
           float friction,
           float kappa);

    void run();
    void runDataCollection();
    void runGNN();

private:
    int n_;
    float dt_;
    int frame_id_;

    Renderer renderer_;
    Physics::PhysicString myString_;
    Measurement::Measurement meas_;
    Estimation::UKF ukf_;
};

}