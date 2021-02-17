#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>

using namespace std;

#define A      0.005
#define ε      0.0001//ε
#define ablity 10
#define parameters 3
#define y_line 2
#define outlayer_nodes 3
#define inlayer_nodes  2+1
#define hiden_layer_nodes  2+1

double database[3][ablity]
{
    {0.3, 0.4, 0.6,0.78,0.96, 1.2,1.13,1.09,1.33, 1.5},//lump-size
    {0.5, 0.6, 0.3, 0.7, 0.4, 0.5, 0.8, 0.9, 0.7,   1},//lump-age
    {  0,   1,   0,   0,   1,   0,   0,   0,   1,   1}
};

double database_A[3][ablity]
{
    {0.3, 0.4, 0.6,0.78,0.96, 1.2,1.13,1.09,1.33, 1.5},//lump-size
    {0.5, 0.6, 0.3, 0.7, 0.4, 0.5, 0.8, 0.9, 0.7,   1},//lump-age
    {  1,   0,   1,   0,   0,   1,   0,   0,   0,   0}
};

double database_B[3][ablity]
{
    {0.3, 0.4, 0.6,0.78,0.96, 1.2,1.13,1.09,1.33, 1.5},//lump-size
    {0.5, 0.6, 0.3, 0.7, 0.4, 0.5, 0.8, 0.9, 0.7,   1},//lump-age
    {  0,   0,   0,   1,   0,   0,   1,   1,   0,   0}
};
//logistic
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

//Neural Networks
//graph composistion
//g(x) = 1/(1+e^(-x))
// inlayer(0)  hiden-layer(1)  outlayer(2)
// a01          a11             a21 
// a02          a12             a22
// a03          a13             a23
//              a14
//类似邻接表
//a01 -> a10 = k<0>11 ----k<n>ij :n为layer i为该层的节点索引 j为下一层的节点索引
//a01 -> a11 = k<0>12
//a01 -> a12 = k<0>13
//a02 -> a10 = k<0>21
//...
//设定a<n,m>(n为layer，m为单层的节点索引) = g(z)
//what the "z": eg->   z<1,1> = a01*k<0>11 + a02*k<0>21 + a03*k<0>31
//即该节点的所有入度的加权和
//so that a<n,m> = g( ∑[ (a<n-1,i>) * (k<n-1><i,m>) ] )
//eg: a12 = g( ∑[ a01*k<0>12 + a02*k<0>22 + a03*k<0>23 ] )
//向量表示：Van = [an1 , an2 , an3 , ... , ans]^T
//上述公式对inlayer除外(不计算第一层--输入层)
//这样我们就能通过给定的输入得到每一层任何一个节点的具体值
//cost function
//back-propagation-algorithm
//设定δ<n,m>(同样除掉第一层-=输入没有误差):为第n层第m个节点的误差值
//eg:   δ21 = a21 - yi<1> --- yi<n> i代表训练特征对应的结果的索引 n在多元分类时有意义，代表本索引结果中对应的元素
//      δ22 = a22 - yi<2>
//      δ23 = a23 - yi<3>
//向量表示：Vδn = [δn1 , δn2 ,δn3 ,..., δns]^T
//And:  δ1 = Vk<1> *  Vδ2 * g'(Vz<1>)
//        __                                             __         
//Vk<n> = | k<n><1,1>  k<n><1,2>  k<n><1,3> ... k<n><1,s> |
//        | k<n><2,1>  k<n><2,2>  k<n><2,3> ... k<n><2,s> | 
//        | k<n><3,1>  k<n><3,2>  k<n><3,3> ... k<n><3,s> | 
//        |   .            .          .     ...           |
//        |   .            .          .     ...           |
//        |   .            .          .     ...           |
//        |_k<n><t,1>       ... ... ...     ... k<n><3,s>_|
//so  
//        __                                 __
//Vk<1> = |  k<n><1,1>  k<n><1,2>  k<n><1,3>  |
//        |  k<n><2,1>  k<n><2,2>  k<n><2,3>  | 
//        |  k<n><3,1>  k<n><3,2>  k<n><3,3>  |
//        |_ k<n><4,1>  k<n><4,2>  k<n><4,3> _| 
//                                          
//Vδ2 =   [ δ21 , δ22 , δ23 ]^T
//g'(Vz<n>) = Van * (1-Van)
//        __                                                              __
//Vz<n> = |  a<n-1,1>*k<n-1>11 + a<n-1,2>*k<n-1>21 +...+ a<n-1,s>*k<n-1>s1 |
//        |  a<n-1,1>*k<n-1>12 + a<n-1,2>*k<n-1>22 +...+ a<n-1,s>*k<n-1>s2 |
//        |  .                         .                        .          |
//        |  .                         .                        .          |
//        |  .                         .                        .          |
//        |_ a<n-1,1>*k<n-1>1t + a<n-1,2>*k<n-1>2t +...+ a<n-1,s>*k<n-1>st_|
//=>g'(Vz<n>) = Van * (1-Van)
//证： g(x) = 1/(1+e^(-x)) => g'(x) = e^(-x)/(1+e^(-x))^2
//取向量中的一个元素
//eg: g'(z<1,1>) = e^( -z<1,1> )/( 1+e^(-z<1,1>) )^2
//    又 a<1,1> = g(z<1,1>) = 1/(1+e^(-z<1,1>)) 
//     1-a<1,1> = 1- g(z<1,1>) = e^(-z<1,1>)/(1+e^(-z<1,1>)) 
//a<1,1> * (1- a<1,1>) equal to g'(z<1,1>)
//所以 g'(Vz<n>) = Van * (1-Van)  
//综上
//for hiden-layer 
//δ<n,m> = ∑(k<n>mi * δ2i) * a<n,m> * (1-a<n,m>)
//Vδn =  Vkn * δ(n+1) * Van * (1-Van)    
        



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

void creat_vy(double **vy)
{
    for(int i=0;i<10;i++)
    {
        vy[i][0] = database[y_line][i];
        vy[i][0+1] = database_A[y_line][i];
        vy[i][0+2] = database_B[y_line][i];
        printf("%0.1lf\t%0.1lf\t%0.1lf\t\r\n",vy[i][0],vy[i][1],vy[i][2]);
    }
}

double g(double x)
{
    return ( 1.0/(1+pow(2.71828,(-1)*x) ) );
}


int main()
{
    double *(vx[ablity]);
    double *(vy[ablity]);
    //第一层的节点
    double inlayer[inlayer_nodes] = {0};
    //第二层的节点
    double hidenlayer[hiden_layer_nodes] = {0};
    //第三层的节点
    double outlayer[outlayer_nodes] = {0};
    //第一层对应下一层每个节点的k
    double k_1[inlayer_nodes][hiden_layer_nodes-1] = {0};
    //第二层对应下一层每个节点的k
    double k_2[hiden_layer_nodes][outlayer_nodes] = {0};
    //只有三层
    //输入函数z
    double z = 0;
    //每层每个节点的δ，计算后可以用来梯度下降
    //第一层的节点
        //输入层没有误差
    //第二层的节点
    double d2[hiden_layer_nodes] = {0};
    double D1[inlayer_nodes][hiden_layer_nodes-1] = {0};
    //第三层的节点
    double d3[outlayer_nodes] = {0};
    double D2[hiden_layer_nodes][outlayer_nodes] = {0};
    //反向传播算法的向量中间值
    double vk_vd[hiden_layer_nodes] = {0};
    //代价J
    double J = 0;

    cout<<"vx:"<<endl;
    for(int i=0;i<10;i++)
        vx[i] = new double(3);
    creat_vx(vx);
    cout<<"vy:"<<endl;
    for(int i=0;i<10;i++)
        vy[i] = new double(3);
    creat_vy(vy);
    //exit(0);
    while(1)
    {
        static long times = 1;
        for(int index=0;index<ablity;index++)
        {
            //1.step
            //give the feature value to the inlayer alternative 
            for(int m=0;m<inlayer_nodes;m++)
                inlayer[m] = vx[index][m];
            //2.step
            //comnpute each node for each layer in each index
            //hiden-layer的计算值
            for(int m=1;m<hiden_layer_nodes;m++)
            {
                for(int i=0;i<inlayer_nodes;i++)
                    z +=  inlayer[i]*k_1[i][m-1];
                hidenlayer[m] = g(z);
                //计算顺序下一个索引的z
                z=0;
            }
            //outlayer的计算值
            hidenlayer[0] = 1;
            for(int m=0;m<outlayer_nodes;m++)
            {
                for(int i=0;i<hiden_layer_nodes;i++)
                    z +=  hidenlayer[i]*k_2[i][m];
                outlayer[m] = g(z);
                //计算顺序下一个索引的z
                z=0;
            }        
            //3.step
            //back-propagation-algorithm
            //compute the δ for each node in each layer
            //先计算输出层的误差值
            for(int i=0;i<outlayer_nodes;i++)
                d3[i] = outlayer[i] - vy[index][i];
            //再计算隐藏层的误差
            for(int i=0;i<hiden_layer_nodes;i++)
            {
                for(int m=0;m<outlayer_nodes;m++)
                {
                    vk_vd[i] += k_2[i][m]*d3[m];
                }
                d2[i] = vk_vd[i]*hidenlayer[i]*(1-hidenlayer[i]);
            }
            //清零vk_vd准备下一次计算
            for(int i=0;i<hiden_layer_nodes;i++)
            {
                vk_vd[i] = 0;
            }
            //4.step
            //算出代价函数的偏导数项(△ij的累计)
            //for layer 2
            for(int i=0;i<inlayer_nodes;i++)
            {
                for(int m=1;m<hiden_layer_nodes;m++)
                {
                    D1[i][m-1] += (inlayer[m]*d2[i]);
                }
            }
            //for layer 3
            for(int i=0;i<hiden_layer_nodes;i++)
            {
                for(int m=0;m<outlayer_nodes;m++)
                {
                    D2[i][m] += (hidenlayer[m]*d3[i]);
                }
            }
            //计算总代价方便调试
            for(int i=0;i<3;i++)
                J += ( (vy[index][i]*log(outlayer[i])+(1-vy[index][i])*log(1-outlayer[i]))*(-1)/ablity );    
        }
        //5.step
        //梯度下降逐层更新
        //for layer 2
        //printf("layer2 k:\r\n");
        for(int i=0;i<inlayer_nodes;i++)
        {
            for(int m=1;m<hiden_layer_nodes;m++)
            {
                k_1[i][m-1] = 0.9997*k_1[i][m-1] - (A * D1[i][m-1])/ablity;//0.997~正则化
                //printf("%0.3lf\t",k_1[i][m]);
            }
            //printf("\n");
        }    
        //for layer 3
        //printf("layer3 k:\r\n");
        for(int i=0;i<hiden_layer_nodes;i++)
        {
            for(int m=0;m<outlayer_nodes;m++)
            {
                k_2[i][m] = 0.9997*k_2[i][m] - (A * D2[i][m])/ablity;
                //printf("%0.3lf\t",k_2[i][m]);
            }
            //printf("\n");
        }
        //清空△ij的累计
        //for layer 2
        for(int i=0;i<inlayer_nodes;i++)
        for(int m=1;m<hiden_layer_nodes;m++)
            D1[i][m-1] = 0;
        //for layer 3
        for(int i=0;i<hiden_layer_nodes;i++)
        for(int m=0;m<outlayer_nodes;m++)
            D2[i][m] = 0;
        //显示代价变化
        //printf("cost:%0.3lf\r\n",J);
        
        //usleep(10*1000);
        times++;
        
        if( (times) == 1000000000)
        {
            printf("cost:%0.3lf\r\n",J);
            break;
        }
        J = 0;//清零代价
    }
    return 0;
}