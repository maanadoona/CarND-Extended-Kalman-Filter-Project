#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() == 0)
    {
        cout << "estimation size should not be zero.\n";
        return rmse;
    }
    if(estimations.size() != ground_truth.size())
    {
        cout << "estimation and ground_truth's size are different.\n";
        return rmse;
    }

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse << rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
    MatrixXd Hj(3,4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //compute the Jacobian matrix
    double px2_py2 = px*px + py*py;
    double sqrt_px2_py2 = sqrt(px2_py2);
    //double pow3_2_px2_py2 = pow(px2_py2, 1.5);
    double pow3_2_px2_py2 = px2_py2*sqrt_px2_py2;

    if(fabs(px2_py2) < 0.0001)
    {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    Hj << px/sqrt_px2_py2,                      py/sqrt_px2_py2,                    0,                  0,
         -py/px2_py2,                           px/px2_py2,                         0,                  0,
          py*(vx*py - vy*px) / pow3_2_px2_py2,  px*(vy*px-vx*py) / pow3_2_px2_py2,  px/sqrt_px2_py2,    py/sqrt_px2_py2;

    return Hj;
}
