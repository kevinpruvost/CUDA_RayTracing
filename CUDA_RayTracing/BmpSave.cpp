#include "BmpSave.h"

#include <iostream>

void BmpSave::SaveBMP(const std::string& filePath, unsigned char* pixelData, int width, int height)
{
    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    fileHeader.bfType = 0x4D42; // 'BM'
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + (width * height * 3);
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height; // Negative height to indicate top-down row order
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = width * height * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    std::ofstream outFile(filePath, std::ios::binary);

    if (!outFile) {
        std::cerr << "Failed to open file for writing: " << filePath << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    outFile.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    outFile.write(reinterpret_cast<char*>(pixelData), width * height * 3);

    outFile.close();

    if (!outFile) {
        std::cerr << "Failed to write file: " << filePath << std::endl;
    }
}
