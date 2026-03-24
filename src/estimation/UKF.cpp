#include "../../include/estimation/UKF.hpp"


namespace Estimation    
    {
        UKF::UKF(Eigen::VectorXd initialState, Physics::PhysicString myString, Measurement::Measurement meas, int n, int N, int measDim, float kappa):   
            state_(initialState), myString_(myString), meas_(meas), n_(n), N_(N), kappa_(kappa)
            {     
                covariance_ = meas_.getInitialCov();        
                processNoise_ = meas_.getProcessNoise();      
                measurementNoise_ = meas_.getMeasureNoise();  

                sigmaPoints_ = Eigen::MatrixXd::Zero(N_, 2*N_+1);
                stateSigma_ = Eigen::MatrixXd::Zero(N_, 2*N_+1);
                measurementSpace_ = Eigen::MatrixXd::Zero(measDim, 2*N_+1);

                w_ = Eigen::VectorXd::Zero(2*N_+1);  
                w_(0) = kappa_/(N_+kappa_);
                for (int i = 1; i < 2*N_+1; i++){
                    w_(i) = 1.0 / (2.0 * (N_ + kappa_));
                }  
            }



        Eigen::MatrixXd UKF::Cholsqrt(const Eigen::MatrixXd& M) {
            Eigen::LLT<Eigen::MatrixXd> llt(M);
            
            if (llt.info() == Eigen::Success) {
                return llt.matrixL();
            }
            Eigen::MatrixXd M_fixed = M + Eigen::MatrixXd::Identity(M.rows(), M.cols()) * 1e-5;
            Eigen::LLT<Eigen::MatrixXd> llt_fixed(M_fixed);
            if (llt_fixed.info() == Eigen::Success) {
                return llt_fixed.matrixL();
            }
            throw std::runtime_error("Cholesky failed even with diagonal loading");
        }


        void UKF::computeSigmaPoints(){

            sigmaPoints_.col(0) = state_;
           
            Eigen::MatrixXd temp = (N_ + kappa_) * covariance_;

            temp = Cholsqrt(temp);

            for (int i = 0; i < N_; i++) {
                sigmaPoints_.col(i + 1)      = state_ + temp.col(i);
                sigmaPoints_.col(i + 1 + N_) = state_ - temp.col(i);
            }
        }



        void UKF::pointsPropagation(float dt){

            for (int i = 0; i < N_*2 +1; i++) {
                myString_.setState(sigmaPoints_.col(i));
                myString_.stepVerlet(dt);
                stateSigma_.col(i) = myString_.getState();
            }
        }


        
        void UKF::predict(float dt){
            computeSigmaPoints();
            pointsPropagation(dt);
            Eigen::VectorXd x_mean = stateSigma_ * w_;
            Eigen::MatrixXd centeredSS = stateSigma_.colwise() - x_mean;
            covariance_ = centeredSS * w_.asDiagonal() * centeredSS.transpose() + processNoise_;
        };



        void UKF::update(const Eigen::VectorXd& trueState){

            //Get real measurement and the measurement space

            measurementSpace_ = meas_.getMeasurementSpace(stateSigma_);

            Eigen::VectorXd z_mean = measurementSpace_ * w_;
            //Eigen::VectorXd z_mean = meas_.computeMeasurementMean(measurementSpace_, w_);
            
            Eigen::MatrixXd centeredMS = meas_.NormalizedCensteredMS(measurementSpace_, z_mean);

            //
            Eigen::MatrixXd Pz = centeredMS * w_.asDiagonal() * centeredMS.transpose() + measurementNoise_;

            //cross-covariance matrix
            Eigen::VectorXd x_mean = stateSigma_ * w_;
            Eigen::MatrixXd centeredSS = stateSigma_.colwise() - x_mean;
            Eigen::MatrixXd Pxz=  centeredSS * w_.asDiagonal() * centeredMS.transpose();

            //Kalman Gain
            Eigen::MatrixXd K = Pxz * Pz.inverse();

            Eigen::VectorXd z_measured = meas_.getRealMeasurement(trueState);
            Eigen::VectorXd innovation = meas_.NormalizedInnovationAngle(z_measured, z_mean);

            //State update
            state_ = state_ + K * innovation;

            //Update covariance
            covariance_ = covariance_ - K * Pz * K.transpose();

        };
    }

