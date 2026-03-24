#include "../../include/Simulation/measurement.hpp"

namespace Measurement{

    Measurement::Measurement(int n, int N): 
        n_(n), N_(N){}


        Eigen::MatrixXd Measurement::getInitialCov(){
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N_, N_)* 1e-3;
            return P;
        }

    
        Eigen::MatrixXd Measurement::getProcessNoise(){

            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N_, N_);

            for (int i = 0; i < n_; i++) {
                Q(i, i)         = 1e-4; 
                Q(n_+i, n_+i)     = 1e-4; 
                Q(2*n_+i, 2*n_+i) = 1e-2; 
                Q(3*n_+i, 3*n_+i) = 1e-2; 
            }
            
            return Q;
        }















// identity measurement model for the test version of the UKF (works)

        Eigen::MatrixXd Measurement::getMeasureNoise(){
            int measDim = 4 * n_; 
            

            Eigen::VectorXd noise_diag(measDim);
            
            for(int i=0; i<n_; i++) {
                noise_diag(i) = 0.01;           // Noise X
                noise_diag(n_+i) = 0.01;        // Noise Y
                noise_diag(2*n_+i) = 0.1;       // Noise Vx (ou OldX)
                noise_diag(3*n_+i) = 0.1;       // Noise Vy (ou OldY)
            }

            return noise_diag.asDiagonal();
        }

        Eigen::VectorXd Measurement::getRealMeasurement(Eigen::VectorXd state){
            return state; 
        }

        Eigen::MatrixXd Measurement::getMeasurementSpace(Eigen::MatrixXd sigmaPoints){
            return sigmaPoints;
        }

        Eigen::VectorXd Measurement::NormalizedInnovationAngle(Eigen::VectorXd z_measured , Eigen::VectorXd z_mean){
            return z_measured - z_mean;
        }

        Eigen::MatrixXd Measurement::NormalizedCensteredMS(Eigen::MatrixXd measurementSpace, Eigen::VectorXd z_mean){
            return measurementSpace.colwise() - z_mean;
        }

















        /*Eigen::MatrixXd Measurement::getMeasureNoise(){

            double sigma_dist   = 0.3;       
            double sigma_angle  = 0.03;       
            double sigma_theta0 = 0.08;       
            int measDim = 4 * (n_ - 1);  
            Eigen::MatrixXd R = Eigen::MatrixXd::Zero(measDim, measDim);

            for (int i = 0; i < n_ - 1; i++)
                R(i, i) = sigma_dist * sigma_dist;

            for (int i = 0; i < n_ - 1; i++)
                R(n_ - 1 + i, n_ - 1 + i) = sigma_angle * sigma_angle;

            R(2 * (n_ - 1), 2 * (n_ - 1)) = sigma_theta0 * sigma_theta0;

            return R;
        }
        
        


        Eigen::VectorXd Measurement::getRealMeasurement(Eigen::VectorXd state)
        {
            int measDim = 4 * (n_ - 1);
            Eigen::VectorXd z(measDim);

            for (int i = 0; i < n_ - 1; i++)
            {
                double dx = state(i) - state(i + 1);
                double dy = state(n_ + i) - state(n_ + i + 1);

                z(i) = std::sqrt(dx*dx + dy*dy);
                z(n_-1+i) = std::atan2(dy, dx);

                double old_dx = state(2*n_+i) - state(2*n_+i+1);
                double old_dy = state(3*n_+i) - state(3*n_+i+1);

                z(2*(n_-1)+i) = std::sqrt(old_dx*old_dx + old_dy*old_dy);
                z(3*(n_-1)+i) = std::atan2(old_dy, old_dx);
            }

            return z;
        }




        Eigen::MatrixXd Measurement::getMeasurementSpace(Eigen::MatrixXd stateSigma)
        {
            int measDim = 4 * (n_ - 1);
            Eigen::MatrixXd zeta(measDim, 2*N_ + 1);

            for (int icol = 0; icol < 2*N_ + 1; icol++)
            {
                for (int i = 0; i < n_ - 1; i++)
                {
                    // ---- current positions
                    double dx  = stateSigma(i, icol)     - stateSigma(i + 1, icol);
                    double dy  = stateSigma(n_ + i, icol) - stateSigma(n_ + i + 1, icol);

                    zeta(i, icol) = std::sqrt(dx*dx + dy*dy);
                    zeta(n_-1+i, icol) = std::atan2(dy, dx);

                    // ---- old positions
                    double old_dx = stateSigma(2*n_ + i, icol) - stateSigma(2*n_ + i + 1, icol);
                    double old_dy = stateSigma(3*n_ + i, icol) - stateSigma(3*n_ + i + 1, icol);

                    zeta(2*(n_-1)+i, icol) = std::sqrt(old_dx*old_dx + old_dy*old_dy);
                    zeta(3*(n_-1)+i, icol) = std::atan2(old_dy, old_dx);
                }
            }

            return zeta;
        }

        
        Eigen::VectorXd Measurement::NormalizedInnovationAngle(Eigen::VectorXd z_measured , Eigen::VectorXd z_mean){

            Eigen::VectorXd innovation = z_measured - z_mean;

            // Normalize angular components
            for (int i = n_-1; i < 2*(n_-1); i++)
                innovation(i) = normalizeAngle(innovation(i));

            for (int i = 3*(n_-1); i < 4*(n_-1); i++)
                innovation(i) = normalizeAngle(innovation(i));
            return innovation;
        }

        Eigen::MatrixXd Measurement::NormalizedCensteredMS(Eigen::MatrixXd measurementSpace, Eigen::VectorXd z_mean){

            Eigen::MatrixXd centeredMS = measurementSpace.colwise() - z_mean;

            // Normalize angular components
            for (int k = 0; k < 2*N_ + 1; k++) {
                for (int i = n_-1; i < 2*(n_-1); i++) {
                    centeredMS(i, k) = normalizeAngle(centeredMS(i, k));
                }
                for (int i = 3*(n_-1); i < 4*(n_-1); i++){
                    centeredMS(i, k) = normalizeAngle(centeredMS(i, k));
                }
            }
            return centeredMS;
        }


        Eigen::VectorXd Measurement::computeMeasurementMean(Eigen::MatrixXd measurementSpace, Eigen::VectorXd w)
        {
            int measDim = measurementSpace.rows();
            int numSigma = measurementSpace.cols();

            Eigen::VectorXd z_mean = Eigen::VectorXd::Zero(measDim);

            for (int k = 0; k < measDim; k++)
            {
                bool isAngle =
                    (k >= (n_-1) && k < 2*(n_-1)) ||     // current angles
                    (k >= 3*(n_-1) && k < 4*(n_-1));     // old angles

                if (!isAngle)
                {
                    // ---- Linear mean (distances)
                    z_mean(k) = measurementSpace.row(k) * w;
                }
                else
                {
                    // ---- Circular mean (angles)
                    double s = 0.0;
                    double c = 0.0;

                    for (int i = 0; i < numSigma; i++)
                    {
                        double theta = measurementSpace(k, i);
                        s += w(i) * std::sin(theta);
                        c += w(i) * std::cos(theta);
                    }

                    z_mean(k) = std::atan2(s, c);
                }
            }

            return z_mean;
        }*/

}





