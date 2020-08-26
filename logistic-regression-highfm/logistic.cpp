#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>

using namespace std;

#define A      0.01
#define ablity 10
#define parameters 3
#define y_line 2

double database[3][ablity]
{
    {0.3, 0.4, 0.6,0.78,0.96, 1.2,1.13,1.09,1.33, 1.5},//lump-size
    {0.5, 0.6, 0.3, 0.7, 0.4, 0.5, 0.8, 0.9, 0.7,   1},//lump-age
    {  0,   1,   0,   1,   1,   0,   1,   0,   1,   1}
};

//1.拟合函数
//h(x) = 1/( 1 + e^[ -(vk * vx) ] )
//(vk * vx) = [ k0, k1, k2, k3,...,kn ] * [ x0, x1, x2,...,xn ]^T(转置)
//          = k0x0 + k1x1 + k2x2 + ... + knxn
//2.代价函数
//J(kn) = - (1/m) * [ ∑[ ylogh(x) + (1-y)log(1-h(x)) ] ]
//3.求导后
//dJ/dk =  [ ∑(h(x) - y)*x ]
//4.更新参数,梯度下降
//k[i] = k[i] - A * [ ∑(h(x) - y)*x ]

void creat_vx(double **vx)
{
    for(int i=0;i<10;i++)
    {
        for(int j=0;j<parameters;j++)
        {
            if(j!=0)
            {
                vx[i][j] = database[j-1][i];
                printf("%0.2lf ",vx[i][j]);
            }
            else
            {
                vx[i][j] = 1;
                printf("%0.2lf ",vx[i][j]);
            }
        }
        cout<<endl;
    }
}

int main()
{
    double *(vx)[10] = {0};
    double fx = 0;
    double hx = 0;
    double j = 0;
    double dj[parameters] = {0};
    double k[parameters] = {0};
    double s_gap[parameters] = {0};

    for(int i=0;i<10;i++)
        vx[i] = (double*)malloc(sizeof(double)*parameters);
    creat_vx(vx);

    while(1)//迭代拟合参数
    {
        static int times = 1;
        for(int index=0;index<ablity;index++)
        {
            //1.构造h(x) = 1/( 1 + e^[ -(vk * vx) ] )
            //(vk * vx) = [ k0, k1, k2, k3,...,kn ] * [ x0, x1, x2,...,xn ]^T(转置)
            //          = k0x0 + k1x1 + k2x2 + ... + knxn
            fx += (k[0] * vx[index][0]);
            for(int i=1;i<parameters;i++)
            {
                fx += (k[i] * vx[index][i]);//一次项
                fx += (k[i] * pow(vx[index][i],2) );//二次项
                fx += (k[i] * vx[index][i]*vx[index][(i+1)%parameters]);        //二次项
            }
            hx = 1/( 1 + pow(2.71828,(-1)*fx) );
            //2.得到偏差和与对应特征值的乘积
            for(int i=0;i<parameters;i++)
            {
                s_gap[i] += ((hx - database[y_line][index])*vx[index][i]);//一次项偏导数
                s_gap[i] += ((hx - database[y_line][index])*pow(vx[index][i],2));//二次项偏导数
                s_gap[i] += ((hx - database[y_line][index])*vx[index][i]*vx[index][(i+1)%parameters]);//二次项偏导数
            }
            //3.得到代价值，方便观察结果
            //j += pow((hx - database[y_line][index]),2)/2/ablity;
            j += (database[y_line][index]*log(hx) + (1-database[y_line][index])*log(1-hx) )/ablity*(-1);
            fx=0;
        }
        //同步更新(可以开线程分别同步跟新)
        for(int i=0;i<parameters;i++)
        {
            dj[i] = A * s_gap[i];         //4.得到代价函数的对于对应特征值的导数值
            k[i]  = k[i] - A * dj[i];     //5.用多元梯度下降算法不断迭代得到最佳参数
        }
        //6.显示整定结果
        for(int i=0;i<parameters;i++)
            printf("k[%d]=%0.3lf\t",i,k[i]);
        cout<<"J(kn):"<<j<<"\tct:"<<times<<endl;
        //7.清零迭代变量，继续迭代
        for(int i=0;i<parameters;i++)
            s_gap[i] = 0;
        j=0;
        //usleep(1000*10);
        times++;
    }

    return 0;
}