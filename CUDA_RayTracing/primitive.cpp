#include"primitive.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "solver.h"

#define ran() ( double( rand() % 32768 ) / 32768 )

const int BEZIER_MAX_DEGREE = 5;
const int Combination[BEZIER_MAX_DEGREE + 1][BEZIER_MAX_DEGREE + 1] =
{	0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    1, 2, 1, 0, 0, 0,
    1, 3, 3, 1, 0, 0,
    1, 4, 6, 4, 1, 0,
    1, 5, 10,10,5, 1
};

const int MAX_COLLIDE_TIMES = 10;
const int MAX_COLLIDE_RANDS = 10;

std::pair<double, double> ExpBlur::GetXY()
{
    double x,y;
    x = ran();
    // x in [0, 1), but with a higher prob to be a small value
    x = pow(2, x)-1;
    y = rand();
    return std::pair<double, double>(x*cos(y),x*sin(y));
}

// ====================================================

Material::Material() {
    color = absor = Color();
    refl = refr = 0;
    diff = spec = 0;
    rindex = 0;
    drefl = 0;
    texture = NULL;
    blur = new ExpBlur();
}

void Material::Input( std::string var , std::stringstream& fin ) {
    if ( var == "color=" ) color.Input( fin );
    if ( var == "absor=" ) absor.Input( fin );
    if ( var == "refl=" ) fin >> refl;
    if ( var == "refr=" ) fin >> refr;
    if ( var == "diff=" ) fin >> diff;
    if ( var == "spec=" ) fin >> spec;
    if ( var == "drefl=" ) fin >> drefl;
    if ( var == "rindex=" ) fin >> rindex;
    if ( var == "texture=" ) {
        std::string file; fin >> file;
        texture = new Bmp;
        texture->Input( file );
    }
    if ( var == "blur=" ) {
        std::string blurname; fin >> blurname;
        if(blurname == "exp")
            blur = new ExpBlur();
    }
}

// ====================================================

Primitive::Primitive() {
    sample = rand();
    material = new Material;
    next = NULL;
}

Primitive::Primitive( const Primitive& primitive ) {
    *this = primitive;
    material = new Material;
    *material = *primitive.material;
}

Primitive::~Primitive() {
    delete material;
}

void Primitive::Input( std::string var , std::stringstream& fin ) {
    material->Input( var , fin );
}

// -----------------------------------------------

Sphere::Sphere() : Primitive() {
    De = Vector3( 0 , 0 , 1 );
    Dc = Vector3( 0 , 1 , 0 );
}

void Sphere::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if (var == "R=")
    {
        fin >> R;

        // AABB for BVH
        min.x = O.x - R;
        max.x = O.x + R;
        min.y = O.y - R;
        max.y = O.y + R;
        min.z = O.z - R;
        max.z = O.z + R;
        centroid = (min + max) / 2;
    }
    if ( var == "De=" ) De.Input( fin );
    if (var == "Dc=")
    {
        Dc.Input(fin);
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Sphere::Collide( Vector3 ray_O , Vector3 ray_V ) {
    ray_V = ray_V.GetUnitVector();
    Vector3 P = ray_O - O;
    double b = -P.Dot( ray_V );
    double det = b * b - P.Module2() + R * R;
    CollidePrimitive ret;

    if ( det > EPS ) {
        det = sqrt( det );
        double x1 = b - det  , x2 = b + det;

        if ( x2 < EPS ) return ret;
        if ( x1 > EPS ) {
            ret.dist = x1;
            ret.front = true;
        } else {
            ret.dist = x2;
            ret.front = false;
        }
    } else {
        return ret;
    }

    ret.C = ray_O + ray_V * ret.dist;   
    ret.N = ( ret.C - O ).GetUnitVector();
    if ( ret.front == false ) ret.N = -ret.N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Sphere::GetTexture(Vector3 crash_C) {
    Vector3 I = ( crash_C - O ).GetUnitVector();
    double a = acos( -I.Dot( De ) );
    double b = acos( std::min( std::max( I.Dot( Dc ) / sin( a ) , -1.0 ) , 1.0 ) );
    double u = a / PI , v = b / 2 / PI;
    if ( I.Dot( Dc * De ) < 0 ) v = 1 - v;
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Plane::Input( std::string var , std::stringstream& fin ) {
    if ( var == "N=" ) N.Input( fin );
    if ( var == "R=" ) fin >> R;
    if ( var == "Dx=" ) Dx.Input( fin );
    if (var == "Dy=")
    {
        Dy.Input(fin);

        // AABB for BVH
        Vector3 refPoint = N * R;
        Vector3 p1 = refPoint + Dx + Dy;
        Vector3 p2 = refPoint - Dx + Dy;
        Vector3 p3 = refPoint + Dx - Dy;
        Vector3 p4 = refPoint - Dx - Dy;
        min.x = std::min(std::min(std::min(p1.x, p2.x), p3.x), p4.x);
        max.x = std::max(std::max(std::max(p1.x, p2.x), p3.x), p4.x);
        min.y = std::min(std::min(std::min(p1.y, p2.y), p3.y), p4.y);
        max.y = std::max(std::max(std::max(p1.y, p2.y), p3.y), p4.y);
        min.z = std::min(std::min(std::min(p1.z, p2.z), p3.z), p4.z);
        max.z = std::max(std::max(std::max(p1.z, p2.z), p3.z), p4.z);
        centroid = (min + max) / 2;
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Plane::Collide( Vector3 ray_O , Vector3 ray_V ) {
    ray_V = ray_V.GetUnitVector();
    N = N.GetUnitVector();
    double d = N.Dot( ray_V );
    CollidePrimitive ret;
    if ( fabs( d ) < EPS ) return ret;
    double l = ( N * R - ray_O ).Dot( N ) / d;
    if ( l < EPS ) return ret;

    ret.dist = l;
    ret.front = ( d < 0 );
    ret.C = ray_O + ray_V * ret.dist;
    ret.N = ( ret.front ) ? N : -N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Plane::GetTexture(Vector3 crash_C) {
    double u = crash_C.Dot( Dx ) / Dx.Module2();
    double v = crash_C.Dot( Dy ) / Dy.Module2();
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------

void Square::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O=" ) O.Input( fin );
    if ( var == "Dx=" ) Dx.Input( fin );
    if (var == "Dy=")
    {
        Dy.Input(fin);

        // AABB for BVH
        min.x = std::min(O.x, O.x + Dx.x + Dy.x);
        max.x = std::max(O.x, O.x + Dx.x + Dy.x);
        min.y = std::min(O.y, O.y + Dx.y + Dy.y);
        max.y = std::max(O.y, O.y + Dx.y + Dy.y);
        min.z = std::min(O.z, O.z + Dx.z + Dy.z);
        max.z = std::max(O.z, O.z + Dx.z + Dy.z);
        centroid = (min + max) / 2;
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Square::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    ray_V = ray_V.GetUnitVector();
    auto N = (Dx * Dy).GetUnitVector();
    double d = N.Dot(ray_V);

    if (fabs(d) < EPS) {
        return ret;
    }

    // solve equation
    double t = (O - ray_O).Dot(N) / d;
    if (t < EPS) {
        return ret;
    }
    auto P = ray_O + ray_V * t;

    // check whether inside square
    double DxLen2 = Dx.Module2();
    double DyLen2 = Dy.Module2();

    double x2 = abs((P - O).Dot(Dx));
    double y2 = abs((P - O).Dot(Dy));
    if (x2 > DxLen2 || y2 > DyLen2) {
        return ret;
    }

    ret.dist = t;
    ret.front = (d < 0);
    ret.C = P;
    ret.N = (ret.front) ? N : -N;
    ret.isCollide = true;
    ret.collide_primitive = this;
    return ret;
}

Color Square::GetTexture(Vector3 crash_C) {
    double u = (crash_C - O).Dot( Dx ) / Dx.Module2() / 2 + 0.5;
    double v = (crash_C - O).Dot( Dy ) / Dy.Module2() / 2 + 0.5;
    return material->texture->GetSmoothColor( u , v );
}

// -----------------------------------------------
// TODO: NEED TO IMPLEMENT. Follow the code given below or totally re-write it
void Cylinder::Input( std::string var , std::stringstream& fin ) {
    if ( var == "O1=" ) O1.Input( fin );
    if ( var == "O2=" ) O2.Input( fin );
    if (var == "R=")
    {
        fin >> R;

        // AABB for BVH
        min.x = std::min(O1.x, O2.x) - R;
        max.x = std::max(O1.x, O2.x) + R;
        min.y = std::min(O1.y, O2.y) - R;
        max.y = std::max(O1.y, O2.y) + R;
        min.z = std::min(O1.z, O2.z) - R;
        max.z = std::max(O1.z, O2.z) + R;
        centroid = (min + max) / 2;
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Cylinder::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    // TODO: NEED TO IMPLEMENT
    return ret;
}

Color Cylinder::GetTexture(Vector3 crash_C) {
    double u = 0.5 ,v = 0.5;

    // TODO: NEED TO IMPLEMENT
    return Color();
}

// -----------------------------------------------

void Bezier::Input( std::string var , std::stringstream& fin ) {
    if (var == "O1=")
    {
        O1.Input(fin);
    }
    if (var == "O2=")
    {
        O2.Input(fin);
        N = (O2 - O1).GetUnitVector();
        Nx = N.GetAnVerticalVector();
        Ny = N * Nx;
    }
    if ( var == "P=" ) {
        degree++;
        double newR, newZ;
        fin>>newZ>>newR;
        if (newR > R_c) R_c = newR;
        R.push_back(newR);
        Z.push_back(newZ);
        
        // AABB for BVH
        min.x = std::min(O1.x, O2.x) - R_c;
        max.x = std::max(O1.x, O2.x) + R_c;
        min.y = std::min(O1.y, O2.y) - R_c;
        max.y = std::max(O1.y, O2.y) + R_c;
        min.z = std::min(O1.z, O2.z) - R_c;
        max.z = std::max(O1.z, O2.z) + R_c;
        centroid = (min + max) / 2;
    }
    Primitive::Input( var , fin );
}

CollidePrimitive Bezier::Collide( Vector3 ray_O , Vector3 ray_V ) {
    CollidePrimitive ret;

    // TODO: NEED TO IMPLEMENT
    return ret;
}

Color Bezier::GetTexture(Vector3 crash_C) {
    double u = 0.5 ,v = 0.5;

    // TODO: NEED TO IMPLEMENT
    return Color();
}

std::pair<double, double> Bezier::valueAt(double u)
{
    return valueAt(u, Z, R);
}


std::pair<double, double> Bezier::valueAt(double u, const std::vector<double>& xs, const std::vector<double>& ys)
{
    const int degree = xs.size() - 1;
    double x = 0;
    double y = 0;
    for (int i = 0; i <= degree; i++) {
        double factor = double(Combination[degree][i]) * pow(u, i) * pow(1 - u, degree - i);
        x += factor * xs[i];
        y += factor * ys[i];
    }
    return std::make_pair(x, y);
}

void Triangle::Input(std::string var, std::stringstream& fin)
{
    if (var == "O1=")
    {
        O1.Input(fin);
    }
    if (var == "O2=")
    {
        O2.Input(fin);
    }
    if (var == "O3=")
    {
        O3.Input(fin);
        N = ((O2 - O1) * (O3 - O1)).GetUnitVector();

        // AABB for BVH
        min.x = std::min(std::min(O1.x, O2.x), O3.x);
        max.x = std::max(std::max(O1.x, O2.x), O3.x);
        min.y = std::min(std::min(O1.y, O2.y), O3.y);
        max.y = std::max(std::max(O1.y, O2.y), O3.y);
        min.z = std::min(std::min(O1.z, O2.z), O3.z);
        max.z = std::max(std::max(O1.z, O2.z), O3.z);
        centroid = (min + max) / 2;
    }
    Primitive::Input(var, fin);
}

CollidePrimitive Triangle::Collide(Vector3 ray_O, Vector3 ray_V)
{
    return CollidePrimitive();
}

Color Triangle::GetTexture(Vector3 crash_C)
{
    return Color();
}
