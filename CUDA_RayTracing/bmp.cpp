#include "bmp.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>

Bmp::Bmp(int H, int W) {
    Initialize(H, W);
}

Bmp::~Bmp() {
    Release();
}

void Bmp::Initialize(int H, int W) {
    strHead.bfReserved1 = 0;
    strHead.bfReserved2 = 0;
    strHead.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    strInfo.biSize = sizeof(BITMAPINFOHEADER);
    strInfo.biPlanes = 1;
    strInfo.biHeight = H;
    strInfo.biWidth = W;
    strInfo.biBitCount = 24;
    strInfo.biCompression = 0;
    strInfo.biSizeImage = ((W * 3 + 3) & ~3) * H;  // Width must be a multiple of 4
    strInfo.biXPelsPerMeter = 0;
    strInfo.biYPelsPerMeter = 0;
    strInfo.biClrUsed = 0;
    strInfo.biClrImportant = 0;

    strHead.bfSize = strInfo.biSizeImage + strHead.bfOffBits;

    ima = new IMAGEDATA * [H];
    for (int i = 0; i < H; i++) {
        ima[i] = new IMAGEDATA[W];
    }
}

void Bmp::Release() {
    if (ima != nullptr) {
        for (int i = 0; i < strInfo.biHeight; i++) {
            delete[] ima[i];
        }
        delete[] ima;
        ima = nullptr;
    }
}

void Bmp::Input(std::string file) {
    Release();
    std::cout << "Loading file: " << file << std::endl;

    FILE* fpi;
    if (fopen_s(&fpi, file.c_str(), "rb") != 0 || fpi == nullptr) {
        std::cerr << "Error opening file: " << file << std::endl;
        return;
    }

    word bfType;
    fread(&bfType, 1, sizeof(word), fpi);
    if (bfType != 0x4D42) { // 'BM' in little-endian
        std::cerr << "Not a BMP file: " << file << std::endl;
        fclose(fpi);
        return;
    }

    fread(&strHead, 1, sizeof(BITMAPFILEHEADER), fpi);
    fread(&strInfo, 1, sizeof(BITMAPINFOHEADER), fpi);

    if (strInfo.biBitCount != 24 || strInfo.biCompression != 0) {
        std::cerr << "Unsupported BMP format: " << file << std::endl;
        fclose(fpi);
        return;
    }

    Initialize(strInfo.biHeight, strInfo.biWidth);

    int paddedRowSize = (strInfo.biWidth * 3 + 3) & ~3;
    byte* rowData = new byte[paddedRowSize];

    for (int i = 0; i < strInfo.biHeight; i++) {
        fread(rowData, 1, paddedRowSize, fpi);
        for (int j = 0; j < strInfo.biWidth; j++) {
            int idx = j * 3;
            ima[i][j].blue = rowData[idx];
            ima[i][j].green = rowData[idx + 1];
            ima[i][j].red = rowData[idx + 2];
        }
    }

    delete[] rowData;
    fclose(fpi);
}

void Bmp::Output(std::string file) {
    FILE* fpw;
    if (fopen_s(&fpw, file.c_str(), "wb") != 0 || fpw == nullptr) {
        std::cerr << "Error opening file: " << file << std::endl;
        return;
    }

    word bfType = 0x4D42;
    fwrite(&bfType, 1, sizeof(word), fpw);
    fwrite(&strHead, 1, sizeof(BITMAPFILEHEADER), fpw);
    fwrite(&strInfo, 1, sizeof(BITMAPINFOHEADER), fpw);

    int paddedRowSize = (strInfo.biWidth * 3 + 3) & ~3;
    byte* rowData = new byte[paddedRowSize];

    for (int i = 0; i < strInfo.biHeight; i++) {
        for (int j = 0; j < strInfo.biWidth; j++) {
            int idx = j * 3;
            rowData[idx] = ima[i][j].blue;
            rowData[idx + 1] = ima[i][j].green;
            rowData[idx + 2] = ima[i][j].red;
        }
        fwrite(rowData, 1, paddedRowSize, fpw);
    }

    delete[] rowData;
    fclose(fpw);
}

void Bmp::SetColor(int i, int j, Color col) {
    ima[i][j].red = static_cast<int>(col.r * 255);
    ima[i][j].green = static_cast<int>(col.g * 255);
    ima[i][j].blue = static_cast<int>(col.b * 255);
}

Color Bmp::GetSmoothColor(double u, double v) {
    double U = (u - floor(u)) * strInfo.biHeight;
    double V = (v - floor(v)) * strInfo.biWidth;
    int U1 = static_cast<int>(floor(U - EPS)), U2 = U1 + 1;
    int V1 = static_cast<int>(floor(V - EPS)), V2 = V1 + 1;
    double rat_U = U2 - U;
    double rat_V = V2 - V;
    if (U1 < 0) U1 = strInfo.biHeight - 1;
    if (U2 == strInfo.biHeight) U2 = 0;
    if (V1 < 0) V1 = strInfo.biWidth - 1;
    if (V2 == strInfo.biWidth) V2 = 0;

    Color ret;
    ret = ret + ima[U1][V1].GetColor() * rat_U * rat_V;
    ret = ret + ima[U1][V2].GetColor() * rat_U * (1 - rat_V);
    ret = ret + ima[U2][V1].GetColor() * (1 - rat_U) * rat_V;
    ret = ret + ima[U2][V2].GetColor() * (1 - rat_U) * (1 - rat_V);
    return ret;
}
