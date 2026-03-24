#pragma once
#include <Eigen/Dense>
#include "../Simulation/PhysicSimulator.hpp"
#include "../Simulation/measurement.hpp"
#include <cmath>
#include <iostream>

namespace Estimation {

    class UKF {
    private:
        Eigen::VectorXd state_;             // State Vector
        Eigen::MatrixXd covariance_;        // Covariance matrix
        Eigen::MatrixXd processNoise_;      // Process noise covariance
        Eigen::MatrixXd measurementNoise_;  // Measurement noise covariance
        Eigen::MatrixXd sigmaPoints_;
        Eigen::MatrixXd stateSigma_;
        Eigen::MatrixXd measurementSpace_;
        Eigen::VectorXd w_;
        Physics::PhysicString myString_;
        Measurement::Measurement meas_;
        int n_, N_;
        float kappa_;


    public:
        UKF(Eigen::VectorXd initialState, Physics::PhysicString myString, Measurement::Measurement meas, int n, int N, int measDim, float kappa);

        Eigen::MatrixXd Cholsqrt(const Eigen::MatrixXd& M);

        void computeSigmaPoints();

        void pointsPropagation(float dt);
        
        void predict( float dt);
        
        void update(const Eigen::VectorXd& trueState);

       
        const Eigen::VectorXd& getState() const { return state_; } //getter
        void setState(const Eigen::VectorXd& s) { state_ = s; } //setter

    };

} 
