//
// Created by Cortuzz on 5/26/2022.
//

#pragma once
#include <utility>
#include <vector>
#include <chrono>
#include <intrin.h>
#include <immintrin.h>
#include <omp.h>

struct PicturesMergingAlgorithm {
protected:
    std::vector<unsigned char> firstPicture;
    std::vector<unsigned char> secondPicture;
    std::vector<unsigned char> resultPicture;
    int size {};

public:
    void setPictures(std::vector<unsigned char> pix1, std::vector<unsigned char> pix2, int _size) {
        firstPicture = std::move(pix1);
        secondPicture = std::move(pix2);
        size = _size;
        resultPicture = std::vector<unsigned char>(size, 0);
    }

    virtual void plusPictures() {
        throw std::exception();
    }

    virtual void minusPictures() {
        throw std::exception();
    }

    virtual void gainExtractPictures() {
        throw std::exception();
    }

    virtual void gainMergePictures() {
        throw std::exception();
    }

    std::vector<unsigned char> getResult() {
        return resultPicture;
    }
};

struct VectorizedPicturesMergingAlgorithm:PicturesMergingAlgorithm {
    const int CHUNK_SIZE = 32;

    void plusPictures() override {
        omp_set_num_threads(16);
        
        int i;
        #pragma omp parallel for private(i) 
        for (i = 0; i < size; i += CHUNK_SIZE) {
            __m256i v1 = _mm256_load_si256((__m256i*) & firstPicture[i]);
            __m256i v2 = _mm256_load_si256((__m256i*) & secondPicture[i]);
            __m256i reg = _mm256_adds_epu8(v1, v2);
            _mm256_store_si256((__m256i*) & resultPicture[i], reg);
        }
    }

    void minusPictures() override {
        for (int i = 0; i < size; i += CHUNK_SIZE) {
            __m256i v1 = _mm256_load_si256((__m256i*)&firstPicture[i]);
            __m256i v2 = _mm256_load_si256((__m256i*)&secondPicture[i]);
            __m256i reg = _mm256_subs_epu8(v1, v2);
            _mm256_store_si256((__m256i*)&resultPicture[i], reg);
        }
    }

    void gainExtractPictures() override {
        throw std::exception();
//        for (int i = 0; i < size; i += CHUNK_SIZE) {
//            __m256i v1 = _mm256_load_si256((__m256i*)&firstPicture[i]);
//            __m256i v2 = _mm256_load_si256((__m256i*)&secondPicture[i]);
//            __m256i reg = _mm256_sub_epi8(_mm256_add_epi8(v1, v2), ...);
//            _mm256_store_si256((__m256i*)&resultPicture[i], reg);
//        }
    }

    void gainMergePictures() override {
        throw std::exception();
//        for (int i = 0; i < size; i += CHUNK_SIZE) {
//            __m256i x = _mm256_load_si256((__m256i*)&firstPicture[i]);
//            __m256i y = _mm256_load_si256((__m256i*)&secondPicture[i]);
//            __m256i z = _mm256_subs_epu8(x, y);
//            _mm256_store_si256((__m256i*)&resultPicture[i], z);
//        }
    }
};
        
struct DefaultPicturesMergingAlgorithm:PicturesMergingAlgorithm {
    void plusPictures() override {
        omp_set_num_threads(16);

        int i;
        #pragma omp parallel for private(i) 
        for (i = 0; i < size; i++) {
            resultPicture[i] = ((firstPicture[i] + secondPicture[i] > 255) ? 255 : firstPicture[i] + secondPicture[i]);
        }
      
    }

    void minusPictures() override {
        for (int i = 0; i < size; i++) {
            (i % 4 != 3) ? resultPicture[i] = ((firstPicture[i] - secondPicture[i] < 0) ? 0 : firstPicture[i] - secondPicture[i]) :
                    resultPicture[i] = 255;
        }
    }

    void gainExtractPictures() override {
        for (int i = 0; i < size; i++) {
            (i % 4 != 3) ? resultPicture[i] = ((firstPicture[i] - secondPicture[i] < -128) ? 0 : firstPicture[i] - secondPicture[i] + 128) :
                    resultPicture[i] = 255;
        }
    }

    void gainMergePictures() override {
        for (int i = 0; i < size; i++) {
            resultPicture[i] = ((firstPicture[i] + secondPicture[i] - 128 > 255) ? 255 : firstPicture[i] + secondPicture[i] - 128);
        }
    }
};
