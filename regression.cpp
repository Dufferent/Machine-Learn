#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>

using namespace std;

#define ablity 10

int database[2][ablity]={
    { 1, 2, 4, 5, 7, 9,15,20,30,40},
    { 3, 8,10,11, 9,13,22,27,40,70}
};
//1.拟合函数-线性回归
//h(x) = k_1*x + k_2;
//2.代价函数
//m为样本数量
//J(k1,k2) = (1/2m)∑[(h_k1_k2(x) - y(x))]^2
//3.梯度算法
//k_n = k_n - dJ(k_1,k_2,...)/dk_n * A
//求导后
//dJ/dk_n = d[(1/2m)∑[(h_k1_k2(x)  - y(x))]^2]/dk_n
//        = d[(1/2m)∑[(k_1*x + k_2 - y(x))]^2]/dk_n
//n=1:    = (1/m)∑[(k_1*x + k_2 - y(x))(x)]
//n=2:    = (1/m)∑[(k_1*x + k_2 - y(x))]

int main()
{
    double k1=0,k2=0;
    double last_k1=0,last_k2=0;
    double hx = 0;
    double J = 0;
    double J1  = 0;
    double J2  = 0;
    double Last_J = 420;
    double D_J = 0;
    int    i  = 0;
    double m  = ablity;
    while(1)
    {
        hx = k1*database[0][i] + k2;
        J += pow((hx - database[1][i]),2);
        J1 += (hx - database[1][i]);
        J2 += ((hx - database[1][i])*database[0][i]);
        i++;
        if(i == ablity)
        {
            static double temp0,temp1;

            //J = J/(2*m);
            //梯度下降算法
            //1.手动求导
            //temp0 = k1 - 0.0001*(J - Last_J)/(k1 - last_k1);
            //temp1 = k2 - 0.0001*(J - Last_J)/(k2 - last_k2);
            //2.求导后直接运算
            temp0 = k1 - 0.0001*J2/m;
            temp1 = k2 - 0.0001*J1/m;

            last_k1 = k1;
            last_k2 = k2;
            /*
            if( (J - Last_J) < 0.0001 && (J - Last_J) > -0.0001 )
                break;
            */
            k1 = temp0;
            k2 = temp1;
            i = 0;
            //Last_J = J;
            printf("k1 = %0.4lf\tk2 = %0.4lf\tJ = %0.4lf\r\n",k1,k2,J/2/m);
            J = 0;
            J1 = 0;
            J2 = 0;
        }
        //usleep(1000);
    }
    return 0;
}