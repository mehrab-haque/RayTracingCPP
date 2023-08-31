#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>
#include<vector>
#include<stack>
#include <cmath>
#include "bitmap_image.hpp"

using namespace std;

double pi = 2*acos(0.0);

double eyeX,eyeY,eyeZ,lookX,lookY,lookZ,upX,upY,upZ,fovY,aspectRatio,near,far,screenWidth,screenHeight,leftLimit=-1,rightLimit=1,topLimit=1,bottomLimit=-1,farZ=1,nearZ=-1,almostInf=1000000;
stack<vector<vector<double>>> transformationStates;
vector<vector<vector<double>>> stage1Triangles,stage2Triangles,stage3Triangles;
double bgColor[3]={0,0,0};
vector<vector<double>> zBuffer;

ifstream sceneFile;
ifstream configFile;
ofstream stage1File;
ofstream stage2File;
ofstream stage3File;
ofstream zBuffFile;



auto getMatrix(int nRows,int nCols,double initialValue){
    vector<vector<double>> outputMatrix( nRows , vector<double> (nCols, initialValue));
    return outputMatrix;
}

auto getVector(int size,double initialValue){
    vector<double> outputVector( size , initialValue);
    return outputVector;
}

auto getIdentityMatrix(int dim){
    auto outputMatrix=getMatrix(dim,dim,0);
    for(int i=0;i<dim;i++)
        outputMatrix[i][i]=1;
    return outputMatrix;
}


void view(vector<vector<double>> matrix){
    for(int i=0;i<matrix.size();i++){
        for(int j=0;j<matrix[0].size();j++)
            cout<<fixed<<setprecision(7)<<matrix[i][j]<<"\t";
        cout<<endl;
    }
}

auto transform(vector<vector<double>>transformMatrix,vector<vector<double>> operandMatrix){
    auto outputMatrix=getMatrix(transformMatrix.size(),operandMatrix[0].size(),1);
    if(transformMatrix[0].size()==operandMatrix.size())
        for(int i=0;i<transformMatrix.size();i++)
            for(int j=0;j<operandMatrix[0].size();j++){
                double sum=0;
                for(int k=0;k<operandMatrix.size();k++)
                    sum+=transformMatrix[i][k]*operandMatrix[k][j];
                outputMatrix[i][j]=sum;
            }
    return outputMatrix;
}

auto copyMatrix(vector<vector<double>>state){
    auto outputMatrix=getMatrix(state.size(),state[0].size(),0);
    for(int i=0;i<state.size();i++)
        for(int j=0;j<state[0].size();j++)
            outputMatrix[i][j]=state[i][j];
    return outputMatrix;
}

auto vectorScalarMultiple(vector<double>x, double m){
    auto outputVector=getVector(x.size(), 0);
    for(int i=0;i<x.size();i++)
        outputVector[i]=m*x[i];
    return outputVector;
}

auto matrixScalarMultiple(vector<vector<double>>matrix, double m){
    auto outputVector=copyMatrix(matrix);
    for(int i=0;i<matrix.size();i++)
        for(int j=0;j<matrix[0].size();j++)
            outputVector[i][j]*=m;
    return outputVector;
}


auto getVector(double x,double y,double z){
    auto outputVector=getVector(3, 0);
    outputVector[0]=x;
    outputVector[1]=y;
    outputVector[2]=z;
    return outputVector;
}

double vectorDotMultiple(vector<double>v1, vector<double>v2){
    double sum=0;
    for(int i=0;i<v1.size();i++)
        sum+=v1[i]*v2[i];
    return sum;
}

auto vectorAddition(vector<double>v1, vector<double>v2){
    auto outputVector=getVector(v1.size(), 0);
    for(int i=0;i<v1.size();i++)
        outputVector[i]=v1[i]+v2[i];
    return outputVector;
}

auto vectorCrossMultiple(vector<double>v1, vector<double>v2){
    auto outputVector=getVector(v1.size(), 0);
    outputVector[0]=v1[1]*v2[2]-v1[2]*v2[1];
    outputVector[1]=v1[2]*v2[0]-v1[0]*v2[2];
    outputVector[2]=v1[0]*v2[1]-v1[1]*v2[0];
    return outputVector;
}

auto getNormalizedVector(vector<double> v){
    double squareSum=0;
    for(int i=0;i<v.size();i++)
        squareSum+=pow(v[i],2);
    double value=sqrt(squareSum);
    return vectorScalarMultiple(v,1/value);
}

auto getDirectionalVector(int direction){
    auto outputVector=getVector(3, 0);
    outputVector[direction]=1;
    return outputVector;
}

//𝑅(𝑥⃗, 𝑎⃗, 𝜃) = cos 𝜃 𝑥⃗ + (1 − cos 𝜃)(𝑎⃗ ∙ 𝑥⃗)𝑎⃗ + sin 𝜃 (𝑎⃗ × 𝑥⃗)
auto rodriguesFunction(vector<double> x,vector<double> a, double thetaRadian){
    auto t_1=vectorScalarMultiple(x,cos(thetaRadian));
    auto t_2=vectorScalarMultiple(a,(1-cos(thetaRadian))*vectorDotMultiple(a,x));
    auto t_3=vectorScalarMultiple(vectorCrossMultiple(a,x),sin(thetaRadian));
    return vectorAddition(t_1,vectorAddition(t_2,t_3));
}

void writeTriangle(ofstream &file,vector<vector<double>> triangle){
        for(int i=0;i<3;i++)
            file<<fixed<<setprecision(7)<<triangle[0][i]<<" "<<triangle[1][i]<<" "<<triangle[2][i]<<endl;
        file<<endl;
}

void updateTransformation(vector<vector<double>> transformationMatrix){
    transformationStates.pop();
    transformationStates.push(transformationMatrix);
}

auto scaleToIdentityWeight(vector<vector<double>> points){
    return matrixScalarMultiple(points,1/points[3][0]);
}

void takeInputAndStage1(){
    if (sceneFile.is_open()){ 
        sceneFile>>eyeX>>eyeY>>eyeZ>>lookX>>lookY>>lookZ>>upX>>upY>>upZ>>fovY>>aspectRatio>>near>>far;
        string input;
        while(true){
            sceneFile>>input;
            if(input=="end")break;
            if(input=="triangle"){
                auto triangleMatrix=getMatrix(4,3,1);
                for(int i=0;i<3;i++)
                    sceneFile>>triangleMatrix[0][i]>>triangleMatrix[1][i]>>triangleMatrix[2][i];
                auto transformedTriangleMatrix=scaleToIdentityWeight(transform(transformationStates.top(),triangleMatrix));
                stage1Triangles.push_back(transformedTriangleMatrix);
                writeTriangle(stage1File,transformedTriangleMatrix);
            }else if(input=="translate"){
                auto translateMatrix=getIdentityMatrix(4);
                for(int i=0;i<3;i++)
                    sceneFile>>translateMatrix[i][3];
                auto newTransformedMatrix=transform(transformationStates.top(),translateMatrix);
                updateTransformation(newTransformedMatrix);
            }
            else if(input=="scale"){
                auto scaleMatrix=getIdentityMatrix(4);
                for(int i=0;i<3;i++)
                    sceneFile>>scaleMatrix[i][i];
                auto newTransformedMatrix=transform(transformationStates.top(),scaleMatrix);
                //view(scaleMatrix);
                updateTransformation(newTransformedMatrix);
            }
            else if(input=="rotate"){
                auto rotationVector=getVector( 3 , 0);
                double angleDegree;
                sceneFile>>angleDegree;
                double angleRadian=angleDegree*pi/180.0;
                for(int i=0;i<3;i++)
                    sceneFile>>rotationVector[i];
                
                rotationVector=getNormalizedVector(rotationVector);
                auto c1=rodriguesFunction(getDirectionalVector(0),rotationVector,angleRadian);
                auto c2=rodriguesFunction(getDirectionalVector(1),rotationVector,angleRadian);
                auto c3=rodriguesFunction(getDirectionalVector(2),rotationVector,angleRadian);

                auto rotationMatrix=getIdentityMatrix(4);

                for(int i=0;i<3;i++){
                    rotationMatrix[i][0]=c1[i];
                    rotationMatrix[i][1]=c2[i];
                    rotationMatrix[i][2]=c3[i];
                }

                auto newTransformedMatrix=transform(transformationStates.top(),rotationMatrix);
                updateTransformation(newTransformedMatrix);
            }
            else if(input=="push"){
                auto newTransformedMatrix=copyMatrix(transformationStates.top());
                transformationStates.push(newTransformedMatrix);
            }
            else if(input=="pop")
                transformationStates.pop();
            
        }
        sceneFile.close(); 
        stage1File.close(); 
    }
}

void executeStage2(){
    auto l=vectorAddition(getVector(lookX,lookY,lookZ),vectorScalarMultiple(getVector(eyeX,eyeY,eyeZ),-1));
    l=getNormalizedVector(l);
    auto r=vectorCrossMultiple(l,getVector(upX,upY,upZ));
    r=getNormalizedVector(r);
    auto u=vectorCrossMultiple(r,l);

    auto tMatrix=getIdentityMatrix(4);

    tMatrix[0][3]=-eyeX;
    tMatrix[1][3]=-eyeY;
    tMatrix[2][3]=-eyeZ;

    auto rMatrix=getIdentityMatrix(4);
    
    for(int i=0;i<3;i++){
        rMatrix[0][i]=r[i];
        rMatrix[1][i]=u[i];
        rMatrix[2][i]=-l[i];
    }

    auto vMatrix=transform(rMatrix,tMatrix);

    for(int i=0;i<stage1Triangles.size();i++){
        auto transformedTriangleMatrix=scaleToIdentityWeight(transform(vMatrix,stage1Triangles[i]));
        stage2Triangles.push_back(transformedTriangleMatrix);
        writeTriangle(stage2File,transformedTriangleMatrix);
    }
    stage2File.close(); 
}


void executeStage3(){
    double fovX=fovY*aspectRatio;
    double t=near*tan(fovY*pi/360);
    double r=near*tan(fovX*pi/360);
    auto pMatrix=getMatrix(4,4,0);
    pMatrix[0][0]=near/r;
    pMatrix[1][1]=near/t;
    pMatrix[2][2]=-(far+near)/(far-near);
    pMatrix[2][3]=-2*far*near/(far-near);
    pMatrix[3][2]=-1;

    for(int i=0;i<stage2Triangles.size();i++){
        auto transformedTriangleMatrix=scaleToIdentityWeight(transform(pMatrix,stage2Triangles[i]));
        stage3Triangles.push_back(transformedTriangleMatrix);
        writeTriangle(stage3File,transformedTriangleMatrix);
    }
    stage3File.close(); 
}

auto getLineZXPlaneIntersectingPoint(vector<double> p1,vector<double> p2,double y){
    double t = (y - p1[1]) / (p2[1] - p1[1]);
    vector<double> intersection;
    intersection.push_back(p1[0] + t * (p2[0] - p1[0]));
    intersection.push_back(y);
    intersection.push_back(p1[2] + t * (p2[2] - p1[2]));
    return intersection;
}

auto isValidPoint(vector<double> point){
    return !isnan(point[0])&&!isinf(point[0])&&!isnan(point[1])&&!isinf(point[1])&&!isnan(point[2])&&!isinf(point[2])&&abs(point[0])<almostInf&&abs(point[1])<almostInf&&abs(point[2])<almostInf;
}

auto getTrianglePLaneIntersections(vector<vector<double>> triangle,double y){
    vector<vector<double>> intersections;
    vector<double> p1,p2,p3;
    for(int i=0;i<3;i++){
        p1.push_back(triangle[i][0]);
        p2.push_back(triangle[i][1]);
        p3.push_back(triangle[i][2]);
    }
    vector<double> intersection=getLineZXPlaneIntersectingPoint(p1,p2,y);
    if(isValidPoint(intersection))intersections.push_back(intersection);
    intersection=getLineZXPlaneIntersectingPoint(p2,p3,y);
    if(isValidPoint(intersection))intersections.push_back(intersection);
    intersection=getLineZXPlaneIntersectingPoint(p3,p1,y);
    if(isValidPoint(intersection))intersections.push_back(intersection);
    return intersections;
}

void printZBuffer(){
    for(int i=0;i<zBuffer.size();i++){
        for(int j=0;j<zBuffer[0].size();j++){
            if(zBuffer[i][j]<farZ)
                zBuffFile<<fixed<<setprecision(6)<<zBuffer[i][j]<<"\t";
        }
        zBuffFile<<endl;
    }
    zBuffFile.close();
}

static unsigned long int g_seed = 1;
inline int randomInt()
{
 g_seed = (214013 * g_seed + 2531011);
 return (g_seed >> 16) & 0x7FFF;
}


void executeStage4(){
    configFile>>screenWidth>>screenHeight;
    bitmap_image image(screenWidth,screenHeight);

    zBuffer=getMatrix(screenHeight,screenWidth,farZ);

    double dX=(rightLimit-leftLimit)/screenWidth;
    double dY=(topLimit-bottomLimit)/screenHeight;

    double topY=topLimit-dY/2;
    double  bottomY=bottomLimit+dY/2;
    double leftX=leftLimit+dX/2;
    double rightX=rightLimit-dX/2;


    for(int i=0;i<stage3Triangles.size();i++){
        auto triangle=stage3Triangles[i];
        int r=randomInt();
        int g=randomInt();
        int b=randomInt();
        double minY=2,maxY=-2;
        for(int i=0;i<3;i++){
            if(triangle[1][i]<minY)minY=triangle[1][i];
            if(triangle[1][i]>maxY)maxY=triangle[1][i];
        }
        if(minY>topY || maxY<bottomY)continue;

        if(minY<bottomY)minY=bottomY;
        if(maxY>topY)maxY=topY;

        double topLineIndex=ceil((topY-maxY)/dY);
        double bottomLineIndex=floor((topY-minY)/dY);

        //cout<<topLineIndex<<", "<<bottomLineIndex<<endl;

        for(int j=topLineIndex;j<=bottomLineIndex;j++){
            double scanLineY=topY-j*dY;
            auto intersections = getTrianglePLaneIntersections(triangle,scanLineY);
            if(intersections[1][0]<intersections[0][0]){
                auto leftPoint=intersections[1];
                intersections[1]=intersections[0];
                intersections[0]=leftPoint;
            }
            if(intersections[0][0]>rightX || intersections[1][0]<leftX)continue;
            double minX=max(intersections[0][0],leftX);
            double maxX=min(intersections[1][0],rightX);

            double leftLineIndex=ceil((minX-leftX)/dX);
            double rightLineIndex=ceil((maxX-leftX)/dX);

            for(int k=leftLineIndex;k<=rightLineIndex;k++){
                double scanPointX=leftX+k*dX;
                double zVal=intersections[0][2]+(intersections[1][2]-intersections[0][2])*(scanPointX-intersections[0][0])/(intersections[1][0]-intersections[0][0]);
                if(zVal<farZ&&zVal>nearZ&&zVal<zBuffer[j][k]){
                    zBuffer[j][k]=zVal;
                    image.set_pixel(k,j,r,g,b);
                }
            }
        }
    }
    configFile.close();
    printZBuffer();
    image.save_image("out.bmp");
}

int main(){
    transformationStates.push(getIdentityMatrix(4));
    sceneFile.open("scene.txt",ios::in); 
    configFile.open("config.txt",ios::in); 
    stage1File.open("stage1.txt"); 
    stage2File.open("stage2.txt");
    stage3File.open("stage3.txt");
    zBuffFile.open("z_buffer.txt");

    //stage1
    takeInputAndStage1();    

    //stage2
    executeStage2();

    //stage3
    executeStage3();

    //stage4
    executeStage4();

    return 0;
}