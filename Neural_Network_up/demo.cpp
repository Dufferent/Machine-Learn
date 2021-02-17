#include "stdio.h"
#include "stdlib.h"
#include "neural.h"
#include "string.h"

void CHECK_PARM(int layers,unsigned int *nodes,double ***model);

int main(int argc,char** argv)
{
    printf ("Neural Network demo-->\r\n");
    /* Demo Test Code */
    int layers;
    unsigned int *nodes;
    double ***model;

    printf  ("请输入网络层数!\r\n");
    scanf   ("%d",&layers);
    nodes = (unsigned int*)malloc(sizeof(unsigned int)*layers);
    for (int i=0;i<layers;i++)
    {
        printf ("请输入第<%d>层的节点数!\r\n",i+1);
        scanf  ("%d",(int*)(&nodes[i]));
    }
    model = (double***)malloc(sizeof(double**)*layers); //第一维:=>层级
    for (int i=0;i<layers;i++)                          //第二维:=>节点
        model[i] = (double**)malloc(sizeof(double*)*nodes[i]);
    for (int i=0;i<layers-1;i++)                        //第三维:=>权重(参数)
        for (int j=0;j<nodes[i];j++)
        {
            model[i][j] = (double*)malloc(sizeof(double)*nodes[i+1]);
            memset(model[i][j],0,sizeof(double)*nodes[i+1]);
        }
    
    Create_Model(layers,nodes,&model);
    // CHECK_PARM(layers,nodes,model);
    return 0;
}
/* 检查每个节点对应下一层节点的训练参数 */
void CHECK_PARM(int layers,unsigned int *nodes,double ***model)
{
    printf  ("DEBUG::训练参数一览!\r\n");
    for (int i=0;i<layers-1;i++)
    {
        for (int j=0;j<nodes[i];j++)
        {
            for (int k=0;k<nodes[i+1];k++)
                printf ("@parm<%d><%d><%d>[%lf]\r\n",i+1,j+1,k+1,model[i][j][k]);
        }
    }
    printf  ("DEBUG::训练参数一览!\r\n");
}