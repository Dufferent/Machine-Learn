#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>

using namespace std;

struct obj{
    double possibility;
    int index;
};



int main()
{
    double fx[3] = {0};
    double x1,x2;
    struct obj my_obj={
        .possibility = 0,
        .index = 0,
    };

    cout<<"please input x1 and x2"<<endl;
    cin>>x1>>x2;

    fx[0] = 1.0/(1 + pow(2.71828,0.291*x2 - 1.568*x1 + 1.711));
    fx[1] = 1.0/(1 + pow(2.71828,1.005*x1 + 12.3*x2 - 6.807));
    fx[2] = 1.0/(1 + pow(2.71828,3.419*x1 - 12.321*x2 + 6.364));

    for(int i=0;i<3;i++)
    {
        if(my_obj.possibility < fx[i])
        {
            my_obj.possibility = fx[i];
            my_obj.index = i;
        }
    }

    cout<<"可能属于第一类 p="<<fx[0]<<endl;
    cout<<"可能属于第A类 p="<<fx[1]<<endl;
    cout<<"可能属于第B类 p="<<fx[2]<<endl;
    /*
    switch (my_obj.index)
    {
        case 0:cout<<"可能属于第一类 p="<<my_obj.possibility<<endl;break;
        case 1:cout<<"可能属于第A类 p="<<my_obj.possibility<<endl;break;
        case 2:cout<<"可能属于第B类 p="<<my_obj.possibility<<endl;break;
    }
    */
    return 0;
}