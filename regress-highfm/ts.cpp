#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>

using namespace std;

int main()
{
    double fx = 0;
    double x[5];
    double k[5] = {-0.116,0.011,-0.002,0.317,0.076};

    cout<<"please input x1 x2 x3 x4"<<endl;
    x[0] = 1;
    cin>>x[1]>>x[2]>>x[3]>>x[4];

    fx += k[0];
    for(int i=1;i<5;i++)
    {
        fx += (k[i] * x[i]);//一次项
        fx += (k[i] * pow(x[i],2) );//二次项
        fx += (k[i] * x[i]*x[(i+1)%5]);//二次项
    }

    printf("对应新特征数据输出预测结果：%0.3lf\n",fx);

    return 0;
}