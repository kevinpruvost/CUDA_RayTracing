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
    if ( var == "R=" ) fin >> R;
    if ( var == "De=" ) De.Input( fin );
    if ( var == "Dc=" ) Dc.Input( fin );
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
    if ( var == "Dy=" ) Dy.Input( fin );
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
    if ( var == "Dy=" ) Dy.Input( fin );
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
    if ( var == "R=" ) fin>>R;
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

void Mesh::Input(std::string var , std::stringstream& fin)
{
    if (var == "O=")
    {
        O.Input(fin);
    }
    if (var == "rotation=")
    {
        rotation.Input(fin);
    }
    if (var == "scale=")
    {
        scale.Input(fin);
    }
    if (var == "file=")
    {
        std::string file;
        fin >> file;
        LoadModel(file);
    }
    Primitive::Input(var, fin);
}

CollidePrimitive Mesh::Collide(Vector3 ray_O, Vector3 ray_V)
{
    return CollidePrimitive();
}

Color Mesh::GetTexture(Vector3 crash_C)
{
    return Color();
}

#include <iomanip>

void Mesh::LoadModel(const std::string& filename)
{
    // Will only be obj files with stricly vertices and faces
    FILE* file;

    if (fopen_s(&file, filename.c_str(), "r") != 0 || file == NULL)
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Compute cos and sin of rotation angles
    double cos_x = cos(rotation.x), sin_x = sin(rotation.x);
    double cos_y = cos(rotation.y), sin_y = sin(rotation.y);
    double cos_z = cos(rotation.z), sin_z = sin(rotation.z);

    char line[256];
    // First get all lines
    std::vector<Vector3> vertices;
    vertices.reserve(50000);
    triangles.reserve(70000);
    while (fgets(line, 256, file))
    {
        std::stringstream ss;
                
        ss << line;
        std::string type;
        ss >> type;
        if (type == "v")
        {
            Vector3 vertex;
            ss >> vertex.x >> vertex.y >> vertex.z;

            // Apply rotation around X axis
            double y_new = cos_x * vertex.y - sin_x * vertex.z;
            double z_new = sin_x * vertex.y + cos_x * vertex.z;
            vertex.y = y_new;
            vertex.z = z_new;

            // Apply rotation around Y axis
            double x_new = cos_y * vertex.x + sin_y * vertex.z;
            z_new = -sin_y * vertex.x + cos_y * vertex.z;
            vertex.x = x_new;
            vertex.z = z_new;

            // Apply rotation around Z axis
            x_new = cos_z * vertex.x - sin_z * vertex.y;
            y_new = sin_z * vertex.x + cos_z * vertex.y;
            vertex.x = x_new;
            vertex.y = y_new;

            vertex.x *= scale.x;
            vertex.y *= scale.y;
            vertex.z *= scale.z;
            vertex.x += O.x;
            vertex.y += O.y;
            vertex.z += O.z;
            min.x = std::min(min.x, vertex.x);
            min.y = std::min(min.y, vertex.y);
            min.z = std::min(min.z, vertex.z);
            max.x = std::max(max.x, vertex.x);
            max.y = std::max(max.y, vertex.y);
            max.z = std::max(max.z, vertex.z);
            vertices.push_back(vertex);
        }
        else if (type == "f")
        {
            std::vector<int> face(3, 0);
            int vertex;
            int i = 0;
            while (ss >> vertex)
            {
                face[i++] = vertex - 1;
            }
            Triangle tri;
            tri.O1 = vertices[face[0]];
            tri.O2 = vertices[face[1]];
            tri.O3 = vertices[face[2]];
            tri.N = ((tri.O2 - tri.O1) * (tri.O3 - tri.O1)).GetUnitVector();

            //Triangle triReverse;
            //triReverse.O1 = vertices[face[0]];
            //triReverse.O2 = vertices[face[2]];
            //triReverse.O3 = vertices[face[1]];
            //triReverse.N = ((triReverse.O2 - triReverse.O1) * (triReverse.O3 - triReverse.O1)).GetUnitVector();
            //triangles.push_back(triReverse);

            triangles.push_back(tri);
        }
    }

    BuildBVH();
}

int CountTriangles(MeshBoundingBox* node) {
    if (node == nullptr) {
        return 0;
    }
    if (node->triangle != nullptr) {
        return 1;
    }
    return CountTriangles(node->left) + CountTriangles(node->right);
}

#include <fstream>
#include <unordered_map>

void CollectTriangles(MeshBoundingBox* node, std::vector<Triangle*>& triangles) {
    if (!node) return;
    if (node->triangle) {
        triangles.push_back(node->triangle);
    }
    else {
        CollectTriangles(node->left, triangles);
        CollectTriangles(node->right, triangles);
    }
}

void WriteOBJ(const std::string& filename, const std::vector<Triangle*>& triangles) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    std::unordered_map<const Vector3 *, int> vertexMap;
    int vertexIndex = 1;

    // Write vertices
    for (const auto& tri : triangles) {
        if (vertexMap.find(&tri->O1) == vertexMap.end()) {
            outFile << "v " << tri->O1.x << " " << tri->O1.y << " " << tri->O1.z << "\n";
            vertexMap[&tri->O1] = vertexIndex++;
        }
        if (vertexMap.find(&tri->O2) == vertexMap.end()) {
            outFile << "v " << tri->O2.x << " " << tri->O2.y << " " << tri->O2.z << "\n";
            vertexMap[&tri->O2] = vertexIndex++;
        }
        if (vertexMap.find(&tri->O3) == vertexMap.end()) {
            outFile << "v " << tri->O3.x << " " << tri->O3.y << " " << tri->O3.z << "\n";
            vertexMap[&tri->O3] = vertexIndex++;
        }
    }

    // Write faces
    for (const auto& tri : triangles) {
        outFile << "f " << vertexMap[&tri->O1] << " " << vertexMap[&tri->O2] << " " << vertexMap[&tri->O3] << "\n";
    }

    outFile.close();
}

void Mesh::BuildBVH()
{
    // Build BVH
    std::vector<Triangle*> tris;
    for (auto& tri : triangles)
    {
        tris.push_back(&tri);
    }
    root = BuildBVHRecursive(tris, 0);

    // verify number of triangles in root
    // int numTris = CountTriangles(root);
    // std::cout << "Number of triangles in root: " << numTris << " vs: " << triangles.size() << std::endl;

        // Rewrite Obj from root to see if everything is still good
    //std::vector<Triangle*> collectedTriangles;
    //CollectTriangles(root, collectedTriangles);
    //WriteOBJ("test.obj", collectedTriangles);
    
}

MeshBoundingBox* Mesh::BuildBVHRecursive(std::vector<Triangle*>& tris, int depth) {
    if (tris.empty()) {
        return nullptr;
    }

    MeshBoundingBox* node = new MeshBoundingBox();

    if (tris.size() == 1) {
        node->triangle = tris[0];
        node->min = Vector3(
            std::min({ tris[0]->O1.x, tris[0]->O2.x, tris[0]->O3.x }),
            std::min({ tris[0]->O1.y, tris[0]->O2.y, tris[0]->O3.y }),
            std::min({ tris[0]->O1.z, tris[0]->O2.z, tris[0]->O3.z })
        );
        node->max = Vector3(
            std::max({ tris[0]->O1.x, tris[0]->O2.x, tris[0]->O3.x }),
            std::max({ tris[0]->O1.y, tris[0]->O2.y, tris[0]->O3.y }),
            std::max({ tris[0]->O1.z, tris[0]->O2.z, tris[0]->O3.z })
        );
        return node;
    }

    // Compute the bounding box for the current set of triangles
    double maxNumericLimit = std::numeric_limits<double>::max();
    Vector3 centroidMin(maxNumericLimit, maxNumericLimit, maxNumericLimit), centroidMax(-maxNumericLimit, -maxNumericLimit, -maxNumericLimit);
    for (auto& tri : tris) {
        Vector3 centroid = (tri->O1 + tri->O2 + tri->O3) / 3.0f;
        centroidMin.x = std::min(centroidMin.x, centroid.x);
        centroidMin.y = std::min(centroidMin.y, centroid.y);
        centroidMin.z = std::min(centroidMin.z, centroid.z);
        centroidMax.x = std::max(centroidMax.x, centroid.x);
        centroidMax.y = std::max(centroidMax.y, centroid.y);
        centroidMax.z = std::max(centroidMax.z, centroid.z);
    }

    // Choose the axis to split on
    int axis = depth % 3;
    std::sort(tris.begin(), tris.end(), [axis](Triangle* a, Triangle* b) {
        Vector3 centroidA = (a->O1 + a->O2 + a->O3) / 3.0f;
        Vector3 centroidB = (b->O1 + b->O2 + b->O3) / 3.0f;
        return centroidA[axis] < centroidB[axis];
    });

    size_t mid = tris.size() / 2;
    std::vector<Triangle*> leftTris(tris.begin(), tris.begin() + mid);
    std::vector<Triangle*> rightTris(tris.begin() + mid, tris.end());

    node->left = BuildBVHRecursive(leftTris, depth + 1);
    node->right = BuildBVHRecursive(rightTris, depth + 1);

    // Update bounding box
    node->min = Vector3(
        std::min(node->left ? node->left->min.x : maxNumericLimit, node->right ? node->right->min.x : maxNumericLimit),
        std::min(node->left ? node->left->min.y : maxNumericLimit, node->right ? node->right->min.y : maxNumericLimit),
        std::min(node->left ? node->left->min.z : maxNumericLimit, node->right ? node->right->min.z : maxNumericLimit)
    );

    node->max = Vector3(
        std::max(node->left ? node->left->max.x : -maxNumericLimit, node->right ? node->right->max.x : -maxNumericLimit),
        std::max(node->left ? node->left->max.y : -maxNumericLimit, node->right ? node->right->max.y : -maxNumericLimit),
        std::max(node->left ? node->left->max.z : -maxNumericLimit, node->right ? node->right->max.z : -maxNumericLimit)
    );

    return node;
}