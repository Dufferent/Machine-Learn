#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>

using namespace std;

#define ablity 10
#define y_line  4
#define A       0.0001
int database[5][ablity]={
    {  1,  2,  4,  5,  7,  9, 15, 20, 30, 40},//x1
    { 10, 30, 38, 60, 80, 87,100,130,160,200},//x2
    {0.7,1.3,1.5,1.9,2.1,2.6,3.0,3.3,3.5,4.0},//x3
    {  4,  5,  7,  8,  9, 10, 11, 12, 14, 15},//x4
    {  3,  8, 10, 11,  9, 13, 22, 27, 40, 70} //y
};

//1.多元的拟合函数
//h(x) = k_0 + k_1*x1 + k_2*x2 + k_3*x3 +...+k_n*xn
//引入向量表示
//                     C,  x1, x2,  x3, x4
//such as: vx_1 = [ 1(ex),  1, 10, 0.7,  4 ] vx_2 = [ 1(ex), 2, 30, 1.5, 5 ]
//         vk   = [    k0, k1, k2,  k3, k4 ]
//so that: h(vx) = vx * vk^(T)
//2.多元代价函数
//J(k_0,k_1,k_2,...,k_n) = (1/2m)∑[(h(x_1,x_2,...x_n) - y)^2]
//引入向量
//J(vk) = (1/2m)∑[(h(vx)-y)^2]
//3.多元梯度下降函数
//vtemp = vk - d[J(vk)]/d(vk_n) * A(学习率)
//求导之后
//dJ(vk)/dvk_n = [ (1/m)∑[(k0 + k1*x1 + k2*x2 +... - y)] , (1/m)∑[(k0 + k1*x1 + k2*x2 +... - y)]*x1 ,... ]
//vtemp = vk - A * [ (1/m)∑[(k0 + k1*x1 + k2*x2 +... - y)] , (1/m)∑[(k0 + k1*x1 + k2*x2 +... - y)]*x1 ,... ]
int main()
{
    double vx[10][5] = {
    //  x0  x1  x2  x3  x4
        {1,  1, 10,0.7,  4},
        {1,  2, 30,1.3,  5},
        {1,  4, 38,1.5,  7},
        {1,  5, 60,1.9,  8},
        {1,  7, 80,2.1,  9},
        {1,  9, 87,2.6, 10},
        {1, 15,100,3.0, 11},
        {1, 20,130,3.3, 12},
        {1, 30,160,3.5, 14},
        {1, 40,200,4.0, 15}
    };
    
    double hx = 0;              //拟合函数
    double s_gap[5] = {0};      //偏差和
    double k[5]  = {0};         //参数池
    double dj[5] = {0};         //代价函数的导数值

    double j = 0;

    while(1)
    {
        for(int index=0;index<ablity;index++)
        {
            //1.构造h(x)= k_0 + k_1*x1 + k_2*x2 + k_3*x3 +...+k_n*xn
            for(int i=0;i<5;i++)
                hx += (k[i] * vx[index][i]);
            for(int i=0;i<5;i++)
                s_gap[i] += ((hx - database[y_line][index])*vx[index][i]);

            j += pow((hx - database[y_line][index]),2)/2/ablity;
            hx=0;
        }
        for(int i=0;i<5;i++)
        {
            dj[i] = (1.0/ablity)*s_gap[i];
            k[i]  = k[i] - A * dj[i]; 
        }

        for(int i=0;i<5;i++)
            printf("k[%d]=%0.3lf\t",i,k[i]);
        cout<<j<<endl;
        for(int i=0;i<5;i++)
            s_gap[i] = 0;

        j=0;
        //usleep(1000*10);
    }

    return 0;
}