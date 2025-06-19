#ifndef KECCAK256_CUH
#define KECCAK256_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// Keccak-256 parameters for Ethereum
#define KECCAK_ROUNDS 24
#define KECCAK_STATE_SIZE 25 // 25 uint64_t (1600 bits)
#define KECCAK_RATE 136      // 1088 bits = 136 bytes (rate for Keccak-256)
#define KECCAK_CAPACITY 64   // 512 bits = 64 bytes
#define KECCAK_HASH_SIZE 32  // 256 bits = 32 bytes

// Define constants for both device and host
__device__ __constant__ uint64_t d_keccak_round_constants[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// Host copy of constants
const uint64_t h_keccak_round_constants[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// ROTL64 operation
__device__ __host__ inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Single-lane indexing
#define index(x, y) ((x) + 5 * (y))

// Keccak permutation function
__device__ __host__ void keccak_permutation(uint64_t state[KECCAK_STATE_SIZE]) {
    uint64_t C[5], D[5], temp;
    int x, y;

    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta step
        for (x = 0; x < 5; x++) {
            C[x] = state[index(x, 0)] ^ state[index(x, 1)] ^ state[index(x, 2)] ^ state[index(x, 3)] ^ state[index(x, 4)];
        }
        for (x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (y = 0; y < 5; y++) {
                state[index(x, y)] ^= D[x];
            }
        }

        // Rho and Pi steps
        temp = state[index(1, 0)];
        x = 1; y = 0;
        for (int t = 0; t < 24; t++) {
            int newX = y;
            int newY = (2 * x + 3 * y) % 5;
            uint64_t temp2 = state[index(newX, newY)];
            state[index(newX, newY)] = rotl64(temp, ((t + 1) * (t + 2) / 2) % 64);
            temp = temp2;
            x = newX;
            y = newY;
        }

        // Chi step
        for (y = 0; y < 5; y++) {
            for (x = 0; x < 5; x++) {
                C[x] = state[index(x, y)];
            }
            for (x = 0; x < 5; x++) {
                state[index(x, y)] = C[x] ^ ((~C[(x + 1) % 5]) & C[(x + 2) % 5]);
            }
        }

        // Iota step
        #ifdef __CUDA_ARCH__
        state[0] ^= d_keccak_round_constants[round];
        #else
        state[0] ^= h_keccak_round_constants[round];
        #endif
    }
}

// Convert little-endian bytes to uint64_t
__device__ __host__ inline uint64_t load_le_u64(const uint8_t* bytes) {
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val |= ((uint64_t)bytes[i]) << (8 * i);
    }
    return val;
}

// Convert uint64_t to little-endian bytes
__device__ __host__ inline void store_le_u64(uint8_t* bytes, uint64_t val) {
    for (int i = 0; i < 8; i++) {
        bytes[i] = (val >> (8 * i)) & 0xFF;
    }
}

// Ethereum's Keccak-256 implementation
__device__ __host__ void keccak256(const unsigned char *input, size_t length, unsigned char *output) {
    uint64_t state[KECCAK_STATE_SIZE] = {0};
    uint8_t temp[KECCAK_RATE];

    // Process all full blocks
    while (length >= KECCAK_RATE) {
        for (size_t i = 0; i < KECCAK_RATE / 8; i++) {
            state[i] ^= load_le_u64(input + i * 8);
        }
        keccak_permutation(state);
        input += KECCAK_RATE;
        length -= KECCAK_RATE;
    }

    // Handle the final block with Ethereum's padding
    memset(temp, 0, KECCAK_RATE);
    memcpy(temp, input, length);

    // Ethereum-specific padding: 0x01 after message, 0x80 at end of block
    temp[length] = 0x01;
    temp[KECCAK_RATE - 1] |= 0x80;

    // XOR the final block
    for (size_t i = 0; i < KECCAK_RATE / 8; i++) {
        state[i] ^= load_le_u64(temp + i * 8);
    }

    keccak_permutation(state);

    // Extract hash (first 32 bytes of state)
    for (size_t i = 0; i < KECCAK_HASH_SIZE / 8; i++) {
        store_le_u64(output + i * 8, state[i]);
    }
}

#endif