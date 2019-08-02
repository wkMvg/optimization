#include <iostream>
#include <Eigen/Dense>
#include <ctime>
using namespace std;

const int kNumObservations = 67;
const double data_[] = {
	0.000000e+00, 1.133898e+00,
	7.500000e-02, 1.334902e+00,
	1.500000e-01, 1.213546e+00,
	2.250000e-01, 1.252016e+00,
	3.000000e-01, 1.392265e+00,
	3.750000e-01, 1.314458e+00,
	4.500000e-01, 1.472541e+00,
	5.250000e-01, 1.536218e+00,
	6.000000e-01, 1.355679e+00,
	6.750000e-01, 1.463566e+00,
	7.500000e-01, 1.490201e+00,
	8.250000e-01, 1.658699e+00,
	9.000000e-01, 1.067574e+00,
	9.750000e-01, 1.464629e+00,
	1.050000e+00, 1.402653e+00,
	1.125000e+00, 1.713141e+00,
	1.200000e+00, 1.527021e+00,
	1.275000e+00, 1.702632e+00,
	1.350000e+00, 1.423899e+00,
	1.425000e+00, 1.543078e+00,
	1.500000e+00, 1.664015e+00,
	1.575000e+00, 1.732484e+00,
	1.650000e+00, 1.543296e+00,
	1.725000e+00, 1.959523e+00,
	1.800000e+00, 1.685132e+00,
	1.875000e+00, 1.951791e+00,
	1.950000e+00, 2.095346e+00,
	2.025000e+00, 2.361460e+00,
	2.100000e+00, 2.169119e+00,
	2.175000e+00, 2.061745e+00,
	2.250000e+00, 2.178641e+00,
	2.325000e+00, 2.104346e+00,
	2.400000e+00, 2.584470e+00,
	2.475000e+00, 1.914158e+00,
	2.550000e+00, 2.368375e+00,
	2.625000e+00, 2.686125e+00,
	2.700000e+00, 2.712395e+00,
	2.775000e+00, 2.499511e+00,
	2.850000e+00, 2.558897e+00,
	2.925000e+00, 2.309154e+00,
	3.000000e+00, 2.869503e+00,
	3.075000e+00, 3.116645e+00,
	3.150000e+00, 3.094907e+00,
	3.225000e+00, 2.471759e+00,
	3.300000e+00, 3.017131e+00,
	3.375000e+00, 3.232381e+00,
	3.450000e+00, 2.944596e+00,
	3.525000e+00, 3.385343e+00,
	3.600000e+00, 3.199826e+00,
	3.675000e+00, 3.423039e+00,
	3.750000e+00, 3.621552e+00,
	3.825000e+00, 3.559255e+00,
	3.900000e+00, 3.530713e+00,
	3.975000e+00, 3.561766e+00,
	4.050000e+00, 3.544574e+00,
	4.125000e+00, 3.867945e+00,
	4.200000e+00, 4.049776e+00,
	4.275000e+00, 3.885601e+00,
	4.350000e+00, 4.110505e+00,
	4.425000e+00, 4.345320e+00,
	4.500000e+00, 4.161241e+00,
	4.575000e+00, 4.363407e+00,
	4.650000e+00, 4.161576e+00,
	4.725000e+00, 4.619728e+00,
	4.800000e+00, 4.737410e+00,
	4.875000e+00, 4.727863e+00,
	4.950000e+00, 4.669206e+00,
};

void gauss_newton(const double m_init, const double c_init, double& m_curr, double& c_curr)
{
	double err_curr = 1000;
	double err_last = 0;

	m_curr = m_init;
	c_curr = c_init;

	size_t iter = 0;

	while (abs(err_curr - err_last) > 0.000001 && iter < 500)
	{
		++iter;
		Eigen::Matrix<double, kNumObservations, 2> jacobin;
		Eigen::Matrix<double, kNumObservations, 1> b;

		for (int i = 0; i < kNumObservations; i++)
		{
			double x = data_[2 * i];
			double y = data_[2 * i + 1];

			jacobin(i, 0) = x * exp(m_curr * x + c_curr);
			jacobin(i, 1) = exp(m_curr * x + c_curr);

			b(i, 0) = exp(m_curr * x + c_curr) - y;
		}

		//直接求逆解线性方程组，还好矩阵维数并不高，否则速度会非常慢
		//Eigen::Matrix<double, 2, 1> delta_m_c = (jacobin.transpose() * jacobin).inverse() * (-jacobin.transpose() * b);
		//Eigen::Matrix<double, 2, 1> delta_m_c = (jacobin.transpose() * jacobin).llt().solve(-jacobin.transpose() * b);
		Eigen::Matrix<double, 2, 1> delta_m_c = (jacobin.transpose() * jacobin).ldlt().solve(-jacobin.transpose() * b);
		//Eigen::Matrix<double, 2, 1> delta_m_c = (jacobin.transpose() * jacobin).lu().solve(-jacobin.transpose() * b);
		//Eigen::Matrix<double, 2, 1> delta_m_c = (jacobin.transpose() * jacobin).householderQr().solve(-jacobin.transpose() *b);

		double m_last = m_curr;
		double c_last = c_curr;

		m_curr = delta_m_c(0, 0) + m_curr;
		c_curr = delta_m_c(1, 0) + c_curr;

		for (int i = 0; i < kNumObservations; i++)
		{
			double x = data_[2 * i];
			double y = data_[2 * i + 1];

			b(i, 0) = exp(m_curr * x + c_curr) - y;
		}
		err_last = err_curr;
		err_curr = b.norm();
		cout << "the" << " " << iter << " " << "error is " << err_curr << endl;
	}
}

/* 步长lamda的选择至关重要，选择太小，则下降速率很慢，运行时间很长，而步长太大，很难到达局部最优，甚至非常大时，就会发散*/
void gradient_descent(const double m_init, const double c_init, double& m_curr, double& c_curr, double lamda)
{
	m_curr = m_init;
	c_curr = c_init;

	double err_last = 100;
	double err_curr = 0;

	size_t iter = 0;

	while (abs(err_curr - err_last) > 0.00000001)
	{
		iter++;
		Eigen::Matrix<double, kNumObservations, 2> jacobin;
		Eigen::Matrix<double, kNumObservations, 1> b;

		//构建jacobin矩阵
		for (int i = 0; i < kNumObservations; i++)
		{
			const double x = data_[2 * i];
			const double y = data_[2 * i + 1];
			jacobin(i, 0) = x * exp(m_curr * x + c_curr);
			jacobin(i, 1) = exp(m_curr * x + c_curr);

			b(i, 0) = exp(m_curr * x + c_curr) - y;
		}

		Eigen::Matrix<double,2,1> delta_m_c = jacobin.transpose() * b;
		
		double m_last = m_curr;
		double c_last = c_curr;

		m_curr -= lamda * delta_m_c(0, 0);
		c_curr -= lamda * delta_m_c(1, 0);

		err_last = err_curr;
		
		for (int i = 0; i < kNumObservations; i++)
		{
			double x = data_[2 * i];
			double y = data_[2 * i + 1];

			b(i) = exp(m_curr * x + c_curr) - y;
		}
		err_curr = b.norm();

		cout << "the" << " " << iter << " " << "error is " << err_curr << endl;
	}
}

// curve formula : y = exp(mx + c);
int main()
{
	double m_init = 0;
	double c_init = 0;

	double m_curr = 0;
	double c_curr = 0;

	//gauss-newton algorithm
	clock_t start_gn = clock();
	gauss_newton(m_init, c_init, m_curr, c_curr);
	clock_t end_gn = clock();
	cout << "total gauss-newton progress cost " << (end_gn - start_gn) << "ms" << endl;

	//gradient descent algorithm
	clock_t start_gd = clock();
	gradient_descent(m_init, c_init, m_curr, c_curr, 0.00001);
	clock_t end_gd = clock();
	cout << "total gradient descent progress cost " << end_gd - start_gd << "ms" << endl;
	while (1);
}

/*
inverse directly
the 1 error is 68.3844
the 2 error is 21.1191
the 3 error is 5.06758
the 4 error is 1.5268
the 5 error is 1.4538
the 6 error is 1.45379
the 7 error is 1.45379
the 8 error is 1.45379
the 9 error is 1.45379
the 10 error is 1.45379
total progress cost 980ms
*/

/*
llt cholesky decomposition
the 1 error is 68.3844
the 2 error is 21.1191
the 3 error is 5.06758
the 4 error is 1.5268
the 5 error is 1.4538
the 6 error is 1.45379
the 7 error is 1.45379
the 8 error is 1.45379
the 9 error is 1.45379
the 10 error is 1.45379
total progress cost 962ms
*/

/*
ldlt chlesky decomposition deformation
the 1 error is 68.3844
the 2 error is 21.1191
the 3 error is 5.06758
the 4 error is 1.5268
the 5 error is 1.4538
the 6 error is 1.45379
the 7 error is 1.45379
the 8 error is 1.45379
the 9 error is 1.45379
the 10 error is 1.45379
total progress cost 953ms
*/

/*
lu decomposition
the 1 error is 68.3844
the 2 error is 21.1191
the 3 error is 5.06758
the 4 error is 1.5268
the 5 error is 1.4538
the 6 error is 1.45379
the 7 error is 1.45379
the 8 error is 1.45379
the 9 error is 1.45379
the 10 error is 1.45379
total progress cost 931ms
*/

/*
qr decomposition
the 1 error is 68.3844
the 2 error is 21.1191
the 3 error is 5.06758
the 4 error is 1.5268
the 5 error is 1.4538
the 6 error is 1.45379
the 7 error is 1.45379
the 8 error is 1.45379
the 9 error is 1.45379
the 10 error is 1.45379
total progress cost 963ms
*/

