#pragma once
#include <Eigen/Dense>
#include "../Simulation/PhysicSimulator.hpp"
#include <cmath>

namespace Measurement{
    class Measurement{
        private:  

        int n_;
        int N_;

        inline double normalizeAngle(double a) {
            while (a > M_PI)  a -= 2*M_PI;
            while (a < -M_PI) a += 2*M_PI;
            return a;
        }

        public:

        Measurement(int n, int N);

        Eigen::MatrixXd getInitialCov();
        Eigen::MatrixXd getProcessNoise();
        Eigen::MatrixXd getMeasureNoise();
        Eigen::VectorXd getRealMeasurement(Eigen::VectorXd state);
        Eigen::MatrixXd getMeasurementSpace(Eigen::MatrixXd z);

        Eigen::VectorXd NormalizedInnovationAngle(Eigen::VectorXd z_measured , Eigen::VectorXd z_mean);
        Eigen::MatrixXd NormalizedCensteredMS(Eigen::MatrixXd measurementSpace, Eigen::VectorXd z_mean);
        Eigen::VectorXd computeMeasurementMean(Eigen::MatrixXd measurementSpace, Eigen::VectorXd w);
        
    };
}