#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>

using namespace std;

#define parameters 3

struct obj{
    double possibility;
    int index;
};



int main()
{
    double fx = 0;
    double k[parameters] = {-1.434,0.218,0.718};
    double g;
    double x[parameters];
    struct obj my_obj={
        .possibility = 0,
        .index = 0,
    };

    cout<<"please input x1 and x2"<<endl;
    cin>>x[1]>>x[2];
    x[0] = 1;//约定俗成

    g += (k[0] * x[0]);
    for(int i=1;i<parameters;i++)
    {
        g += (k[i] * x[i]);//一次项
        g += (k[i] * pow(x[i],2) );//二次项
        g += (k[i] * x[i]*x[(i+1)%parameters]);        //二次项
    }

    fx = 1/( 1 + pow(2.71828,(-1)*g) );

    if(fx > 0.5)
        cout<<"maybe a positive lump!p="<<fx<<endl;
    else
        cout<<"maybe a negetive lump!p="<<fx<<endl;
    

    return 0;
}