#include "cpuminer-config.h"
#include "miner.h"

#include <string.h>
#include <stdint.h>

#ifdef USE_SHA1_OPENSSL
#include <openssl/sha.h>
#endif

#ifdef USE_SHA1_NEON
#include <arm_neon.h>
#endif

#ifdef USE_SHA1_SSE2
#include <emmintrin.h>
#endif

#ifdef __XOP__
#include <x86intrin.h>
#endif

#ifdef USE_SHA1_AVX2
#include <immintrin.h>
#endif

#ifdef USE_SHA1_PIQPU
#include "/opt/vc/src/hello_pi/hello_fft/mailbox.h"

#define GPU_MEM_FLG     0xC
#define GPU_MEM_MAP     0x0
#define MAX_CODE_SIZE   8192
#define VPM_SIZE		256
#define QPUTHR			16
#define NUM_QPUS        12
#define UNIFORMS		13
struct memory_map {
    unsigned int code[MAX_CODE_SIZE];
    unsigned int uniforms[NUM_QPUS][UNIFORMS];		// 13 parameters per QPU
    unsigned int msg[NUM_QPUS][2];
	unsigned int input[NUM_QPUS][VPM_SIZE];			// input buffer for the QPU
    unsigned int results[NUM_QPUS][VPM_SIZE];		// result buffer for the QPU
};

unsigned size;
unsigned handle;
void *arm_ptr;
int mb;
struct memory_map *arm_map;
unsigned vc_msg;

unsigned int qpu_code[] = {
#include "sha1_raspiqpu.hex"
};

int qpu_ini();
void qpu_end();
#endif


// constants and initial values defined in SHA-1
#define K0 0x5A827999
#define K1 0x6ED9EBA1
#define K2 0x8F1BBCDC
#define K3 0xCA62C1D6

#define H0 0x67452301
#define H1 0xEFCDAB89
#define H2 0x98BADCFE
#define H3 0x10325476
#define H4 0xC3D2E1F0

#define ROL32(_val32, _nBits) (((_val32)<<(_nBits))|((_val32)>>(32-(_nBits))))
#define Ch(x,y,z) ((x&(y^z))^z)
#define Maj(x,y,z) (((x|y)&z)|(x&y))

// W[t] = ROL32(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1);
#define SHABLK(t) (W[t&15] = ROL32(W[(t+13)&15] ^ W[(t+8)&15] ^ W[(t+2)&15] ^ W[t&15], 1))

#define _RS0(v,w,x,y,z,i) { z += Ch(w,x,y) + i + K0 + ROL32(v,5);  w=ROL32(w,30); }
#define _RS00(v,w,x,y,z)  { z += Ch(w,x,y) + K0 + ROL32(v,5);  w=ROL32(w,30); }
#define _RS1(v,w,x,y,z,i) { z += (w^x^y) + i + K1 + ROL32(v,5);  w=ROL32(w,30); }

#define _R0(v,w,x,y,z,t) { z += Ch(w,x,y) + SHABLK(t) + K0 + ROL32(v,5);  w=ROL32(w,30); }
#define _R1(v,w,x,y,z,t) { z += (w^x^y) + SHABLK(t) + K1 + ROL32(v,5);  w=ROL32(w,30); }
#define _R2(v,w,x,y,z,t) { z += Maj(w,x,y) + SHABLK(t) + K2 + ROL32(v,5);  w=ROL32(w,30); }
#define _R3(v,w,x,y,z,t) { z += (w^x^y) + SHABLK(t) + K3 + ROL32(v,5);  w=ROL32(w,30); }


void sha1hash12byte(const char *input, uint32_t *m_state)
{
	uint32_t W[16];
	uint32_t a, b, c, d, e;
	int i;

	// SHA-1 initialization constants
	m_state[0] = H0;
	m_state[1] = H1;
	m_state[2] = H2;
	m_state[3] = H3;
	m_state[4] = H4;

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	// input[0] to input[11], 12byte, 96bits
	for (i = 0; i < 3; i++){
		W[i] = input[4*i+0] << 24 | input[4*i+1] << 16 | input[4*i+2] << 8 | input[4*i+3];
	}

	W[3] = 0x80000000;		// padding

/*
	for (int i = 4; i < 15; i++){
		W[i] = 0;
	}
*/

	W[15] = 12 * 8;		// bits of Message Block (12 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS00(b, c, d, e, a);		// W[4] == 0
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	W[0] = ROL32(W[2] ^ W[0], 1);		// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	_RS0(e, a, b, c, d, W[0]);

	W[1] = ROL32(W[3] ^ W[1], 1);		// (17, W[14]==0, W[9]==0, W[3], W[1])
	_RS0(d, e, a, b, c, W[1]);

	W[2] = ROL32(W[15] ^ W[2], 1);		// (18, W[15], W[10]==0, W[4]==0, W[2])
	_RS0(c, d, e, a, b, W[2]);

	W[3] = ROL32(W[0] ^ W[3], 1);		// (19, W[0], W[11]==0, W[5]==0, W[3])
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	W[4] = ROL32(W[1], 1);				// (20, W[1], W[12]==0, W[6]==0, W[4]==0)
	_RS1(a, b, c, d, e, W[4]);

	W[5] = ROL32(W[2], 1);				// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	_RS1(e, a, b, c, d, W[5]);

	W[6] = ROL32(W[3], 1);				// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	_RS1(d, e, a, b, c, W[6]);

	W[7] = ROL32(W[4] ^ W[15], 1);		// (23, W[4], W[15], W[9]==0, W[7]==0)
	_RS1(c, d, e, a, b, W[7]);

	W[8] = ROL32(W[5] ^ W[0], 1);		// (24, W[5], W[0], W[10]==0, W[8]==0)
	_RS1(b, c, d, e, a, W[8]);

	W[9] = ROL32(W[6] ^ W[1], 1);		// (25, W[6], W[1], W[11]==0, W[9]==0)
	_RS1(a, b, c, d, e, W[9]);

	W[10] = ROL32(W[7] ^ W[2], 1);		// (26, W[7], W[2], W[12]==0, W[10]==0)
	_RS1(e, a, b, c, d, W[10]);

	W[11] = ROL32(W[8] ^ W[3], 1);		// (27, W[8], W[3], W[13]==0, W[11]==0)
	_RS1(d, e, a, b, c, W[11]);

	W[12] = ROL32(W[9] ^ W[4], 1);		// (28, W[9], W[4], W[14]==0, W[12]==0)
	_RS1(c, d, e, a, b, W[12]);

	W[13] = ROL32(W[10] ^ W[5] ^ W[15], 1);		// (29, W[10], W[5], W[15], W[13]==0)
	_RS1(b, c, d, e, a, W[13]);

	W[14] = ROL32(W[11] ^ W[6] ^ W[0], 1);		// (30, W[11], W[6], W[0], W[14]==0)
	_RS1(a, b, c, d, e, W[14]);

	W[15] = ROL32(W[12] ^ W[7] ^ W[1] ^ W[15], 1);		// (31, W[12], W[7], W[1], W[15])
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;

	m_state[0] = swab32(m_state[0]);
	m_state[1] = swab32(m_state[1]);
	m_state[2] = swab32(m_state[2]);
	m_state[3] = swab32(m_state[3]);
	m_state[4] = swab32(m_state[4]);
}


void sha1hash80byte(const uint8_t *input, uint32_t *m_state)
{
	uint32_t W[16];
	uint32_t a, b, c, d, e;
	int i;

	// SHA-1 initialization constants
	m_state[0] = H0;
	m_state[1] = H1;
	m_state[2] = H2;
	m_state[3] = H3;
	m_state[4] = H4;

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	// input[0] to input[63], 64bytes, 512bits
	for (i = 0; i < 16; i++){
		W[i] = input[4*i+0] << 24 | input[4*i+1] << 16 | input[4*i+2] << 8 | input[4*i+3];
	}

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS0(a, b, c, d, e, W[5]);
	_RS0(e, a, b, c, d, W[6]);
	_RS0(d, e, a, b, c, W[7]);
	_RS0(c, d, e, a, b, W[8]);
	_RS0(b, c, d, e, a, W[9]);
	_RS0(a, b, c, d, e, W[10]);
	_RS0(e, a, b, c, d, W[11]);
	_RS0(d, e, a, b, c, W[12]);
	_RS0(c, d, e, a, b, W[13]);
	_RS0(b, c, d, e, a, W[14]);
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	_R0(e, a, b, c, d, 16);
	_R0(d, e, a, b, c, 17);
	_R0(c, d, e, a, b, 18);
	_R0(b, c, d, e, a, 19);

	// round 20 to 39
	_R1(a, b, c, d, e, 20);
	_R1(e, a, b, c, d, 21);
	_R1(d, e, a, b, c, 22);
	_R1(c, d, e, a, b, 23);
	_R1(b, c, d, e, a, 24);
	_R1(a, b, c, d, e, 25);
	_R1(e, a, b, c, d, 26);
	_R1(d, e, a, b, c, 27);
	_R1(c, d, e, a, b, 28);
	_R1(b, c, d, e, a, 29);
	_R1(a, b, c, d, e, 30);
	_R1(e, a, b, c, d, 31);
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	// input[64] to input[79], 16bytes, 128bits
	for (i = 0; i < 4; i++){
		W[i] = input[4*i+64] << 24 | input[4*i+65] << 16 | input[4*i+66] << 8 | input[4*i+67];
	}

	W[4] = 0x80000000;		// padding

/*
	for (int i = 5; i < 15; i++){
		W[i] = 0;
	}
*/

	W[15] = 80 * 8;		// bits of Message Block (80 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	W[0] = ROL32(W[2] ^ W[0], 1);		// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	_RS0(e, a, b, c, d, W[0]);

	W[1] = ROL32(W[3] ^ W[1], 1);		// (17, W[14]==0, W[9]==0, W[3], W[1])
	_RS0(d, e, a, b, c, W[1]);

	W[2] = ROL32(W[15] ^ W[4] ^ W[2], 1);		// (18, W[15], W[10]==0, W[4], W[2])
	_RS0(c, d, e, a, b, W[2]);

	W[3] = ROL32(W[0] ^ W[3], 1);		// (19, W[0], W[11]==0, W[5]==0, W[3])
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	W[4] = ROL32(W[1] ^ W[4], 1);		// (20, W[1], W[12]==0, W[6]==0, W[4])
	_RS1(a, b, c, d, e, W[4]);

	W[5] = ROL32(W[2], 1);				// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	_RS1(e, a, b, c, d, W[5]);

	W[6] = ROL32(W[3], 1);				// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	_RS1(d, e, a, b, c, W[6]);

	W[7] = ROL32(W[4] ^ W[15], 1);		// (23, W[4], W[15], W[9]==0, W[7]==0)
	_RS1(c, d, e, a, b, W[7]);

	W[8] = ROL32(W[5] ^ W[0], 1);		// (24, W[5], W[0], W[10]==0, W[8]==0)
	_RS1(b, c, d, e, a, W[8]);

	W[9] = ROL32(W[6] ^ W[1], 1);		// (25, W[6], W[1], W[11]==0, W[9]==0)
	_RS1(a, b, c, d, e, W[9]);

	W[10] = ROL32(W[7] ^ W[2], 1);		// (26, W[7], W[2], W[12]==0, W[10]==0)
	_RS1(e, a, b, c, d, W[10]);

	W[11] = ROL32(W[8] ^ W[3], 1);		// (27, W[8], W[3], W[13]==0, W[11]==0)
	_RS1(d, e, a, b, c, W[11]);

	W[12] = ROL32(W[9] ^ W[4], 1);		// (28, W[9], W[4], W[14]==0, W[12]==0)
	_RS1(c, d, e, a, b, W[12]);

	W[13] = ROL32(W[10] ^ W[5] ^ W[15], 1);		// (29, W[10], W[5], W[15], W[13]==0)
	_RS1(b, c, d, e, a, W[13]);

	W[14] = ROL32(W[11] ^ W[6] ^ W[0], 1);		// (30, W[11], W[6], W[0], W[14]==0)
	_RS1(a, b, c, d, e, W[14]);

	W[15] = ROL32(W[12] ^ W[7] ^ W[1] ^ W[15], 1);		// (31, W[12], W[7], W[1], W[15])
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;

	m_state[0] = swab32(m_state[0]);
	m_state[1] = swab32(m_state[1]);
	m_state[2] = swab32(m_state[2]);
	m_state[3] = swab32(m_state[3]);
	m_state[4] = swab32(m_state[4]);
}


void b64enc(const uint32_t *hash, char *str)
{
	const char b64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

	str[0] = b64t[hash[0] >> 26];
	str[1] = b64t[(hash[0] >> 20) & 63];
	str[2] = b64t[(hash[0] >> 14) & 63];
	str[3] = b64t[(hash[0] >> 8) & 63];
	str[4] = b64t[(hash[0] >> 2) & 63];
	str[5] = b64t[(hash[0] << 4 | hash[1] >> 28) & 63];
	str[6] = b64t[(hash[1] >> 22) & 63];
	str[7] = b64t[(hash[1] >> 16) & 63];
	str[8] = b64t[(hash[1] >> 10) & 63];
	str[9] = b64t[(hash[1] >> 4) & 63];
	str[10] = b64t[(hash[1] << 2 | hash[2] >> 30) & 63];
	str[11] = b64t[(hash[2] >> 24) & 63];
	str[12] = b64t[(hash[2] >> 18) & 63];
	str[13] = b64t[(hash[2] >> 12) & 63];
	str[14] = b64t[(hash[2] >> 6) & 63];
	str[15] = b64t[hash[2] & 63];
	str[16] = b64t[hash[3] >> 26];
	str[17] = b64t[(hash[3] >> 20) & 63];
	str[18] = b64t[(hash[3] >> 14) & 63];
	str[19] = b64t[(hash[3] >> 8) & 63];
	str[20] = b64t[(hash[3] >> 2) & 63];
	str[21] = b64t[(hash[3] << 4 | hash[4] >> 28) & 63];
	str[22] = b64t[(hash[4] >> 22) & 63];
	str[23] = b64t[(hash[4] >> 16) & 63];
	str[24] = b64t[(hash[4] >> 10) & 63];
	str[25] = b64t[(hash[4] >> 4) & 63];
	str[26] = b64t[(hash[4] << 2) & 63];
	str[27] = 0;
}


#ifdef USE_SHA1_OPT		//////////////////////////////////////////////////

// input - array of Little Endian uint32_t, 16elements, 64bytes
void sha1hash80byte_1st(const uint32_t *input, uint32_t *m_state)
{
	uint32_t W[16];
	uint32_t a, b, c, d, e;

	// SHA-1 initialization constants
	m_state[0] = H0;
	m_state[1] = H1;
	m_state[2] = H2;
	m_state[3] = H3;
	m_state[4] = H4;

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	memcpy(W, input, 64);

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS0(a, b, c, d, e, W[5]);
	_RS0(e, a, b, c, d, W[6]);
	_RS0(d, e, a, b, c, W[7]);
	_RS0(c, d, e, a, b, W[8]);
	_RS0(b, c, d, e, a, W[9]);
	_RS0(a, b, c, d, e, W[10]);
	_RS0(e, a, b, c, d, W[11]);
	_RS0(d, e, a, b, c, W[12]);
	_RS0(c, d, e, a, b, W[13]);
	_RS0(b, c, d, e, a, W[14]);
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	_R0(e, a, b, c, d, 16);
	_R0(d, e, a, b, c, 17);
	_R0(c, d, e, a, b, 18);
	_R0(b, c, d, e, a, 19);

	// round 20 to 39
	_R1(a, b, c, d, e, 20);
	_R1(e, a, b, c, d, 21);
	_R1(d, e, a, b, c, 22);
	_R1(c, d, e, a, b, 23);
	_R1(b, c, d, e, a, 24);
	_R1(a, b, c, d, e, 25);
	_R1(e, a, b, c, d, 26);
	_R1(d, e, a, b, c, 27);
	_R1(c, d, e, a, b, 28);
	_R1(b, c, d, e, a, 29);
	_R1(a, b, c, d, e, 30);
	_R1(e, a, b, c, d, 31);
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;
}


void sha1hash80byte_2nd_opt(const uint32_t *input, const uint32_t *prehash, char *str)
{
	uint32_t W[16];
	uint32_t a, b, c, d, e;
	uint32_t m_state[5];
	int i;

	const char b64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

	for (i = 0; i < 5; i++){
		m_state[i] = prehash[i];
	}

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	memcpy(W, input, 16);

	W[4] = 0x80000000;		// padding

	W[15] = 80 * 8;		// bits of Message Block (80 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	W[0] = ROL32(W[2] ^ W[0], 1);		// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	_RS0(e, a, b, c, d, W[0]);

	W[1] = ROL32(W[3] ^ W[1], 1);		// (17, W[14]==0, W[9]==0, W[3], W[1])
	_RS0(d, e, a, b, c, W[1]);

	W[2] = ROL32(W[15] ^ W[4] ^ W[2], 1);		// (18, W[15], W[10]==0, W[4], W[2])
	_RS0(c, d, e, a, b, W[2]);

	W[3] = ROL32(W[0] ^ W[3], 1);		// (19, W[0], W[11]==0, W[5]==0, W[3])
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	W[4] = ROL32(W[1] ^ W[4], 1);		// (20, W[1], W[12]==0, W[6]==0, W[4])
	_RS1(a, b, c, d, e, W[4]);

	W[5] = ROL32(W[2], 1);				// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	_RS1(e, a, b, c, d, W[5]);

	W[6] = ROL32(W[3], 1);				// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	_RS1(d, e, a, b, c, W[6]);

	W[7] = ROL32(W[4] ^ W[15], 1);		// (23, W[4], W[15], W[9]==0, W[7]==0)
	_RS1(c, d, e, a, b, W[7]);

	W[8] = ROL32(W[5] ^ W[0], 1);		// (24, W[5], W[0], W[10]==0, W[8]==0)
	_RS1(b, c, d, e, a, W[8]);

	W[9] = ROL32(W[6] ^ W[1], 1);		// (25, W[6], W[1], W[11]==0, W[9]==0)
	_RS1(a, b, c, d, e, W[9]);

	W[10] = ROL32(W[7] ^ W[2], 1);		// (26, W[7], W[2], W[12]==0, W[10]==0)
	_RS1(e, a, b, c, d, W[10]);

	W[11] = ROL32(W[8] ^ W[3], 1);		// (27, W[8], W[3], W[13]==0, W[11]==0)
	_RS1(d, e, a, b, c, W[11]);

	W[12] = ROL32(W[9] ^ W[4], 1);		// (28, W[9], W[4], W[14]==0, W[12]==0)
	_RS1(c, d, e, a, b, W[12]);

	W[13] = ROL32(W[10] ^ W[5] ^ W[15], 1);		// (29, W[10], W[5], W[15], W[13]==0)
	_RS1(b, c, d, e, a, W[13]);

	W[14] = ROL32(W[11] ^ W[6] ^ W[0], 1);		// (30, W[11], W[6], W[0], W[14]==0)
	_RS1(a, b, c, d, e, W[14]);

	W[15] = ROL32(W[12] ^ W[7] ^ W[1] ^ W[15], 1);		// (31, W[12], W[7], W[1], W[15])
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;

	// Base64 encode
	str[0] = b64t[m_state[0] >> 26];
	str[1] = b64t[(m_state[0] >> 20) & 63];
	str[2] = b64t[(m_state[0] >> 14) & 63];
	str[3] = b64t[(m_state[0] >> 8) & 63];
	str[4] = b64t[(m_state[0] >> 2) & 63];
	str[5] = b64t[(m_state[0] << 4 | m_state[1] >> 28) & 63];
	str[6] = b64t[(m_state[1] >> 22) & 63];
	str[7] = b64t[(m_state[1] >> 16) & 63];
	str[8] = b64t[(m_state[1] >> 10) & 63];
	str[9] = b64t[(m_state[1] >> 4) & 63];
	str[10] = b64t[(m_state[1] << 2 | m_state[2] >> 30) & 63];
	str[11] = b64t[(m_state[2] >> 24) & 63];
	str[12] = b64t[(m_state[2] >> 18) & 63];
	str[13] = b64t[(m_state[2] >> 12) & 63];
	str[14] = b64t[(m_state[2] >> 6) & 63];
	str[15] = b64t[m_state[2] & 63];
	str[16] = b64t[m_state[3] >> 26];
	str[17] = b64t[(m_state[3] >> 20) & 63];
	str[18] = b64t[(m_state[3] >> 14) & 63];
	str[19] = b64t[(m_state[3] >> 8) & 63];
	str[20] = b64t[(m_state[3] >> 2) & 63];
	str[21] = b64t[(m_state[3] << 4 | m_state[4] >> 28) & 63];
	str[22] = b64t[(m_state[4] >> 22) & 63];
	str[23] = b64t[(m_state[4] >> 16) & 63];
	str[24] = b64t[(m_state[4] >> 10) & 63];
	str[25] = b64t[(m_state[4] >> 4) & 63];

	memcpy(str + 26, str, 11);
}


void sha1hash12byte_opt(const char *input, uint32_t *hash)
{
	uint32_t W[16];
	uint32_t a, b, c, d, e;
	uint32_t m_state[5];
	int i;
	char trip[13], tripkey[13];

	const char trip64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/'
	};

	// SHA-1 initialization constants
	m_state[0] = H0;
	m_state[1] = H1;
	m_state[2] = H2;
	m_state[3] = H3;
	m_state[4] = H4;

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 3; i++){
		W[i] = input[4*i+0] << 24 | input[4*i+1] << 16 | input[4*i+2] << 8 | input[4*i+3];
	}

	W[3] = 0x80000000;		// padding

	W[15] = 12 * 8;		// bits of Message Block (12 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS00(b, c, d, e, a);		// W[4] == 0
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	W[0] = ROL32(W[2] ^ W[0], 1);		// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	_RS0(e, a, b, c, d, W[0]);

	W[1] = ROL32(W[3] ^ W[1], 1);		// (17, W[14]==0, W[9]==0, W[3], W[1])
	_RS0(d, e, a, b, c, W[1]);

	W[2] = ROL32(W[15] ^ W[2], 1);		// (18, W[15], W[10]==0, W[4]==0, W[2])
	_RS0(c, d, e, a, b, W[2]);

	W[3] = ROL32(W[0] ^ W[3], 1);		// (19, W[0], W[11]==0, W[5]==0, W[3])
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	W[4] = ROL32(W[1], 1);				// (20, W[1], W[12]==0, W[6]==0, W[4]==0)
	_RS1(a, b, c, d, e, W[4]);

	W[5] = ROL32(W[2], 1);				// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	_RS1(e, a, b, c, d, W[5]);

	W[6] = ROL32(W[3], 1);				// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	_RS1(d, e, a, b, c, W[6]);

	W[7] = ROL32(W[4] ^ W[15], 1);		// (23, W[4], W[15], W[9]==0, W[7]==0)
	_RS1(c, d, e, a, b, W[7]);

	W[8] = ROL32(W[5] ^ W[0], 1);		// (24, W[5], W[0], W[10]==0, W[8]==0)
	_RS1(b, c, d, e, a, W[8]);

	W[9] = ROL32(W[6] ^ W[1], 1);		// (25, W[6], W[1], W[11]==0, W[9]==0)
	_RS1(a, b, c, d, e, W[9]);

	W[10] = ROL32(W[7] ^ W[2], 1);		// (26, W[7], W[2], W[12]==0, W[10]==0)
	_RS1(e, a, b, c, d, W[10]);

	W[11] = ROL32(W[8] ^ W[3], 1);		// (27, W[8], W[3], W[13]==0, W[11]==0)
	_RS1(d, e, a, b, c, W[11]);

	W[12] = ROL32(W[9] ^ W[4], 1);		// (28, W[9], W[4], W[14]==0, W[12]==0)
	_RS1(c, d, e, a, b, W[12]);

	W[13] = ROL32(W[10] ^ W[5] ^ W[15], 1);		// (29, W[10], W[5], W[15], W[13]==0)
	_RS1(b, c, d, e, a, W[13]);

	W[14] = ROL32(W[11] ^ W[6] ^ W[0], 1);		// (30, W[11], W[6], W[0], W[14]==0)
	_RS1(a, b, c, d, e, W[14]);

	W[15] = ROL32(W[12] ^ W[7] ^ W[1] ^ W[15], 1);		// (31, W[12], W[7], W[1], W[15])
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] += a;
	m_state[1] += b;
	m_state[2] += c;
	m_state[3] += d;
	m_state[4] += e;

	// trip test
	if (m_state[0] >> 2 == trip_target_uint){
		trip[0] = trip64t[m_state[0] >> 26];
		trip[1] = trip64t[(m_state[0] >> 20) & 63];
		trip[2] = trip64t[(m_state[0] >> 14) & 63];
		trip[3] = trip64t[(m_state[0] >> 8) & 63];
		trip[4] = trip64t[(m_state[0] >> 2) & 63];
		trip[5] = trip64t[(m_state[0] << 4 | m_state[1] >> 28) & 63];
		trip[6] = trip64t[(m_state[1] >> 22) & 63];
		trip[7] = trip64t[(m_state[1] >> 16) & 63];
		trip[8] = trip64t[(m_state[1] >> 10) & 63];
		trip[9] = trip64t[(m_state[1] >> 4) & 63];
		trip[10] = trip64t[(m_state[1] << 2 | m_state[2] >> 30) & 63];
		trip[11] = trip64t[(m_state[2] >> 24) & 63];
		trip[12] = 0;

		if (! strncmp(opt_findtrip, trip, strlen(opt_findtrip))){
			memcpy(tripkey, input, 12);
			tripkey[12] = 0;

			applog(LOG_INFO, "tripkey: #%s, trip: %s (yay!!!)", tripkey, trip);

			fprintf(fp_trip, "%s\t#%s\n", trip, tripkey);
			fflush(fp_trip);
		}
	}

	for (i = 0; i < 5; i++){
		hash[i] ^= m_state[i];
	}
}

#endif	// USE_SHA1_OPT		//////////////////////////////////////////////////


#ifdef USE_SHA1_NEON		//////////////////////////////////////////////////

#define MM_OR(a, b) vorrq_u32((a), (b))
#define MM_AND(a, b) vandq_u32((a), (b))
#define MM_XOR(a, b) veorq_u32((a), (b))
#define MM_ADD(a, b) vaddq_u32((a), (b))

#define MM_SLLI(a, b) vshlq_n_u32((a), (b))
#define MM_SRLI(a, b) vshrq_n_u32((a), (b))

#define MM_SET1(a) vdupq_n_u32((a))

#undef ROL32
#undef Ch
#undef Maj

#define ROL32(x, i) vsliq_n_u32(vshrq_n_u32((x), 32-(i)), (x), (i))
#define Ch(x, y, z) vbslq_u32((x), (y), (z))
#define Maj(x, y, z) Ch(MM_XOR((x), (z)), (y), (z))

#undef SHABLK
#define SHABLK(t) (W[(t)&15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[((t)+13)&15], W[((t)+8)&15]), W[((t)+2)&15]), W[(t)&15]), 1))

#undef _RS0
#define _RS0(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Ch(w,x,y), (i)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS00
#define _RS00(v,w,x,y,z) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(Ch(w,x,y), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS1
#define _RS1(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), (i)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R0
#define _R0(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Ch(w,x,y), SHABLK(t)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R1
#define _R1(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R2
#define _R2(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Maj(w,x,y), SHABLK(t)), MM_SET1(K2)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R3
#define _R3(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K3)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}


void sha1hash12byte_neon(const char *input, uint32x4_t *hash)
{
	__attribute__((aligned(16))) uint32x4_t W[16];
	__attribute__((aligned(16))) uint32x4_t a, b, c, d, e;
	__attribute__((aligned(16))) uint32x4_t m_state[5];
	int i, j;
	char trip[13], tripkey[13];
	__attribute__((aligned(16))) uint32_t tmp0[4], tmp1[4], tmp2[4];

	const char trip64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/'
	};

	// SHA-1 initialization constants
	m_state[0] = MM_SET1(H0);
	m_state[1] = MM_SET1(H1);
	m_state[2] = MM_SET1(H2);
	m_state[3] = MM_SET1(H3);
	m_state[4] = MM_SET1(H4);

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 3; i++){
		W[i] = *(uint32x4_t*)(&input[16 * i]);
	}

	W[3] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(12 * 8);		// bits of Message Block (12 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS00(b, c, d, e, a);		// W[4] == 0
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4]==0, W[2])
	W[2] = ROL32(MM_XOR(W[15], W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4]==0)
	W[4] = ROL32(W[1], 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// trip test
	*(uint32x4_t *)tmp0 = m_state[0];

	for (i = 0; i < 4; i++){
		if (tmp0[i] >> 2 == trip_target_uint){
			*(uint32x4_t *)tmp1 = m_state[1];
			*(uint32x4_t *)tmp2 = m_state[2];

			trip[0] = trip64t[tmp0[i] >> 26];
			trip[1] = trip64t[(tmp0[i] >> 20) & 63];
			trip[2] = trip64t[(tmp0[i] >> 14) & 63];
			trip[3] = trip64t[(tmp0[i] >> 8) & 63];
			trip[4] = trip64t[(tmp0[i] >> 2) & 63];
			trip[5] = trip64t[(tmp0[i] << 4 | tmp1[i] >> 28) & 63];
			trip[6] = trip64t[(tmp1[i] >> 22) & 63];
			trip[7] = trip64t[(tmp1[i] >> 16) & 63];
			trip[8] = trip64t[(tmp1[i] >> 10) & 63];
			trip[9] = trip64t[(tmp1[i] >> 4) & 63];
			trip[10] = trip64t[(tmp1[i] << 2 | tmp2[i] >> 30) & 63];
			trip[11] = trip64t[(tmp2[i] >> 24) & 63];
			trip[12] = 0;

			if (! strncmp(opt_findtrip, trip, strlen(opt_findtrip))){
				for (j = 0; j < 3; j++){
					tripkey[0 + 4 * j] = input[3 + 16 * j + 4 * i];
					tripkey[1 + 4 * j] = input[2 + 16 * j + 4 * i];
					tripkey[2 + 4 * j] = input[1 + 16 * j + 4 * i];
					tripkey[3 + 4 * j] = input[0 + 16 * j + 4 * i];
				}
				tripkey[12] = 0;

				applog(LOG_INFO, "tripkey: #%s, trip: %s (yay!!!)", tripkey, trip);

				fprintf(fp_trip, "%s\t#%s\n", trip, tripkey);
				fflush(fp_trip);
			}
		}
	}

	for (i = 0; i < 5; i++){
		hash[i] = MM_XOR(hash[i], m_state[i]);
	}
}


void sha1hash80byte_2nd_neon(const uint32_t *input, const uint32_t *prehash, char *str)
{
	__attribute__((aligned(16))) uint32x4_t W[16];
	__attribute__((aligned(16))) uint32x4_t a, b, c, d, e;
	__attribute__((aligned(16))) uint32x4_t m_state[5];
	int i;
	__attribute__((aligned(16))) uint32_t tmp[4];

	const char b64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

	for (i = 0; i < 5; i++){
		m_state[i] = MM_SET1(prehash[i]);
	}

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 4; i++){
		W[i] = *(uint32x4_t *)(&input[4 * i]);
	}

	W[4] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(80 * 8);		// bits of Message Block (80 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4], W[2])
	W[2] = ROL32(MM_XOR(MM_XOR(W[15], W[4]), W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4])
	W[4] = ROL32(MM_XOR(W[1], W[4]), 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// Base64 encode
#define _B64TL(i) { \
	str[(i)] = b64t[tmp[0]]; \
	str[(i)+48*1] = b64t[tmp[1]]; \
	str[(i)+48*2] = b64t[tmp[2]]; \
	str[(i)+48*3] = b64t[tmp[3]]; \
}

	// str[0] = b64t[hash[0] >> 26];
	*(uint32x4_t *)tmp = MM_SRLI(m_state[0], 26);
	_B64TL(0);

	// str[1] = b64t[(hash[0] >> 20) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[0], 20), MM_SET1(63));
	_B64TL(1);

	// str[2] = b64t[(hash[0] >> 14) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[0], 14), MM_SET1(63));
	_B64TL(2);

	// str[3] = b64t[(hash[0] >> 8) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[0], 8), MM_SET1(63));
	_B64TL(3);

	// str[4] = b64t[(hash[0] >> 2) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[0], 2), MM_SET1(63));
	_B64TL(4);

	// str[5] = b64t[(hash[0] << 4 | hash[1] >> 28) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_OR(MM_SLLI(m_state[0], 4), MM_SRLI(m_state[1], 28)), MM_SET1(63));
	_B64TL(5);

	// str[6] = b64t[(hash[1] >> 22) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[1], 22), MM_SET1(63));
	_B64TL(6);

	// str[7] = b64t[(hash[1] >> 16) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[1], 16), MM_SET1(63));
	_B64TL(7);

	// str[8] = b64t[(hash[1] >> 10) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[1], 10), MM_SET1(63));
	_B64TL(8);

	// str[9] = b64t[(hash[1] >> 4) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[1], 4), MM_SET1(63));
	_B64TL(9);

	// str[10] = b64t[(hash[1] << 2 | hash[2] >> 30) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_OR(MM_SLLI(m_state[1], 2), MM_SRLI(m_state[2], 30)), MM_SET1(63));
	_B64TL(10);

	// str[11] = b64t[(hash[2] >> 24) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[2], 24), MM_SET1(63));
	_B64TL(11);

	// str[12] = b64t[(hash[2] >> 18) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[2], 18), MM_SET1(63));
	_B64TL(12);

	// str[13] = b64t[(hash[2] >> 12) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[2], 12), MM_SET1(63));
	_B64TL(13);

	// str[14] = b64t[(hash[2] >> 6) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[2], 6), MM_SET1(63));
	_B64TL(14);

	// str[15] = b64t[hash[2] & 63];
	*(uint32x4_t *)tmp = MM_AND(m_state[2], MM_SET1(63));
	_B64TL(15);

	// str[16] = b64t[hash[3] >> 26];
	*(uint32x4_t *)tmp = MM_SRLI(m_state[3], 26);
	_B64TL(16);

	// str[17] = b64t[(hash[3] >> 20) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[3], 20), MM_SET1(63));
	_B64TL(17);

	// str[18] = b64t[(hash[3] >> 14) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[3], 14), MM_SET1(63));
	_B64TL(18);

	// str[19] = b64t[(hash[3] >> 8) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[3], 8), MM_SET1(63));
	_B64TL(19);

	// str[20] = b64t[(hash[3] >> 2) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[3], 2), MM_SET1(63));
	_B64TL(20);

	// str[21] = b64t[(hash[3] << 4 | hash[4] >> 28) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_OR(MM_SLLI(m_state[3], 4), MM_SRLI(m_state[4], 28)), MM_SET1(63));
	_B64TL(21);

	// str[22] = b64t[(hash[4] >> 22) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[4], 22), MM_SET1(63));
	_B64TL(22);

	// str[23] = b64t[(hash[4] >> 16) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[4], 16), MM_SET1(63));
	_B64TL(23);

	// str[24] = b64t[(hash[4] >> 10) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[4], 10), MM_SET1(63));
	_B64TL(24);

	// str[25] = b64t[(hash[4] >> 4) & 63];
	*(uint32x4_t *)tmp = MM_AND(MM_SRLI(m_state[4], 4), MM_SET1(63));
	_B64TL(25);

#undef _B64TL

	for (i = 0; i < 4; i++){
		memcpy(str + 26 + 48 * i, str + 48 * i, 11);
	}
}


#endif	// USE_SHA1_NEON	//////////////////////////////////////////////////


#ifdef USE_SHA1_SSE2	//////////////////////////////////////////////////

#define MM_OR(a, b) _mm_or_si128((a), (b))
#define MM_AND(a, b) _mm_and_si128((a), (b))
#define MM_XOR(a, b) _mm_xor_si128((a), (b))
#define MM_ADD(a, b) _mm_add_epi32((a), (b))

#define MM_SLLI(a, b) _mm_slli_epi32((a), (b))
#define MM_SRLI(a, b) _mm_srli_epi32((a), (b))

#define MM_SET1(a) _mm_set1_epi32((a))

#define MM_LOAD(a) _mm_load_si128((__m128i *)(a))
#define MM_STORE(a, b) _mm_store_si128((__m128i *)(a), (b))

#undef ROL32
#undef Ch
#undef Maj

#ifdef __XOP__
#define ROL32(_val32, _nBits) _mm_roti_epi32((_val32), (_nBits))
#define Ch(x,y,z) _mm_cmov_si128((y),(z),(x))
#define Maj(x,y,z) Ch(MM_XOR((x),(z)),(y),(z))
#else
#define ROL32(_val32, _nBits) (MM_OR(MM_SLLI((_val32), (_nBits)), MM_SRLI((_val32), 32-(_nBits))))
#define Ch(x,y,z) (MM_XOR(MM_AND((x), MM_XOR((y), (z))), (z)))
#define Maj(x,y,z) (MM_OR(MM_AND(MM_OR((x), (y)), (z)), MM_AND((x), (y))))
#endif

#undef SHABLK
#define SHABLK(t) (W[(t)&15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[((t)+13)&15], W[((t)+8)&15]), W[((t)+2)&15]), W[(t)&15]), 1))

#undef _RS0
#define _RS0(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Ch(w,x,y), (i)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS00
#define _RS00(v,w,x,y,z) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(Ch(w,x,y), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS1
#define _RS1(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), (i)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R0
#define _R0(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Ch(w,x,y), SHABLK(t)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R1
#define _R1(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R2
#define _R2(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(Maj(w,x,y), SHABLK(t)), MM_SET1(K2)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R3
#define _R3(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K3)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}


void sha1hash12byte_sse2(const char *input, __m128i *hash)
{
	__attribute__((aligned(16))) __m128i W[16];
	__attribute__((aligned(16))) __m128i a, b, c, d, e;
	__attribute__((aligned(16))) __m128i m_state[5];
	int i, j;
	char trip[13], tripkey[13];
	__attribute__((aligned(16))) uint32_t tmp0[4], tmp1[4], tmp2[4];

	const char trip64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/'
	};

	// SHA-1 initialization constants
	m_state[0] = MM_SET1(H0);
	m_state[1] = MM_SET1(H1);
	m_state[2] = MM_SET1(H2);
	m_state[3] = MM_SET1(H3);
	m_state[4] = MM_SET1(H4);

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 3; i++){
		W[i] = MM_LOAD(&input[16 * i]);
	}

	W[3] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(12 * 8);		// bits of Message Block (12 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS00(b, c, d, e, a);		// W[4] == 0
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4]==0, W[2])
	W[2] = ROL32(MM_XOR(W[15], W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4]==0)
	W[4] = ROL32(W[1], 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// trip test
	MM_STORE(tmp0, m_state[0]);

	for (i = 0; i < 4; i++){
		if (tmp0[i] >> 2 == trip_target_uint){
			MM_STORE(tmp1, m_state[1]);
			MM_STORE(tmp2, m_state[2]);

			trip[0] = trip64t[tmp0[i] >> 26];
			trip[1] = trip64t[(tmp0[i] >> 20) & 63];
			trip[2] = trip64t[(tmp0[i] >> 14) & 63];
			trip[3] = trip64t[(tmp0[i] >> 8) & 63];
			trip[4] = trip64t[(tmp0[i] >> 2) & 63];
			trip[5] = trip64t[(tmp0[i] << 4 | tmp1[i] >> 28) & 63];
			trip[6] = trip64t[(tmp1[i] >> 22) & 63];
			trip[7] = trip64t[(tmp1[i] >> 16) & 63];
			trip[8] = trip64t[(tmp1[i] >> 10) & 63];
			trip[9] = trip64t[(tmp1[i] >> 4) & 63];
			trip[10] = trip64t[(tmp1[i] << 2 | tmp2[i] >> 30) & 63];
			trip[11] = trip64t[(tmp2[i] >> 24) & 63];
			trip[12] = 0;

			if (! strncmp(opt_findtrip, trip, strlen(opt_findtrip))){
				for (j = 0; j < 3; j++){
					tripkey[0 + 4 * j] = input[3 + 16 * j + 4 * i];
					tripkey[1 + 4 * j] = input[2 + 16 * j + 4 * i];
					tripkey[2 + 4 * j] = input[1 + 16 * j + 4 * i];
					tripkey[3 + 4 * j] = input[0 + 16 * j + 4 * i];
				}
				tripkey[12] = 0;

				applog(LOG_INFO, "tripkey: #%s, trip: %s (yay!!!)", tripkey, trip);

				fprintf(fp_trip, "%s\t#%s\n", trip, tripkey);
				fflush(fp_trip);
			}
		}
	}

	for (i = 0; i < 5; i++){
		hash[i] = MM_XOR(hash[i], m_state[i]);
	}
}


void sha1hash80byte_2nd_sse2(const uint32_t *input, const uint32_t *prehash, char *str)
{
	__attribute__((aligned(16))) __m128i W[16];
	__attribute__((aligned(16))) __m128i a, b, c, d, e;
	__attribute__((aligned(16))) __m128i m_state[5];
	int i;
	__attribute__((aligned(16))) uint32_t tmp[4];

	const char b64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

	for (i = 0; i < 5; i++){
		m_state[i] = MM_SET1(prehash[i]);
	}

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 4; i++){
		W[i] = MM_LOAD(&input[4 * i]);
	}

	W[4] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(80 * 8);		// bits of Message Block (80 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4], W[2])
	W[2] = ROL32(MM_XOR(MM_XOR(W[15], W[4]), W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4])
	W[4] = ROL32(MM_XOR(W[1], W[4]), 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// Base64 encode
#define _B64TL(i) { \
	str[(i)] = b64t[tmp[0]]; \
	str[(i)+48*1] = b64t[tmp[1]]; \
	str[(i)+48*2] = b64t[tmp[2]]; \
	str[(i)+48*3] = b64t[tmp[3]]; \
}

	// str[0] = b64t[hash[0] >> 26];
	MM_STORE(tmp, MM_SRLI(m_state[0], 26));
	_B64TL(0);

	// str[1] = b64t[(hash[0] >> 20) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 20), MM_SET1(63)));
	_B64TL(1);

	// str[2] = b64t[(hash[0] >> 14) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 14), MM_SET1(63)));
	_B64TL(2);

	// str[3] = b64t[(hash[0] >> 8) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 8), MM_SET1(63)));
	_B64TL(3);

	// str[4] = b64t[(hash[0] >> 2) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 2), MM_SET1(63)));
	_B64TL(4);

	// str[5] = b64t[(hash[0] << 4 | hash[1] >> 28) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[0], 4), MM_SRLI(m_state[1], 28)), MM_SET1(63)));
	_B64TL(5);

	// str[6] = b64t[(hash[1] >> 22) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 22), MM_SET1(63)));
	_B64TL(6);

	// str[7] = b64t[(hash[1] >> 16) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 16), MM_SET1(63)));
	_B64TL(7);

	// str[8] = b64t[(hash[1] >> 10) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 10), MM_SET1(63)));
	_B64TL(8);

	// str[9] = b64t[(hash[1] >> 4) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 4), MM_SET1(63)));
	_B64TL(9);

	// str[10] = b64t[(hash[1] << 2 | hash[2] >> 30) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[1], 2), MM_SRLI(m_state[2], 30)), MM_SET1(63)));
	_B64TL(10);

	// str[11] = b64t[(hash[2] >> 24) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 24), MM_SET1(63)));
	_B64TL(11);

	// str[12] = b64t[(hash[2] >> 18) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 18), MM_SET1(63)));
	_B64TL(12);

	// str[13] = b64t[(hash[2] >> 12) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 12), MM_SET1(63)));
	_B64TL(13);

	// str[14] = b64t[(hash[2] >> 6) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 6), MM_SET1(63)));
	_B64TL(14);

	// str[15] = b64t[hash[2] & 63];
	MM_STORE(tmp, MM_AND(m_state[2], MM_SET1(63)));
	_B64TL(15);

	// str[16] = b64t[hash[3] >> 26];
	MM_STORE(tmp, MM_SRLI(m_state[3], 26));
	_B64TL(16);

	// str[17] = b64t[(hash[3] >> 20) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 20), MM_SET1(63)));
	_B64TL(17);

	// str[18] = b64t[(hash[3] >> 14) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 14), MM_SET1(63)));
	_B64TL(18);

	// str[19] = b64t[(hash[3] >> 8) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 8), MM_SET1(63)));
	_B64TL(19);

	// str[20] = b64t[(hash[3] >> 2) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 2), MM_SET1(63)));
	_B64TL(20);

	// str[21] = b64t[(hash[3] << 4 | hash[4] >> 28) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[3], 4), MM_SRLI(m_state[4], 28)), MM_SET1(63)));
	_B64TL(21);

	// str[22] = b64t[(hash[4] >> 22) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 22), MM_SET1(63)));
	_B64TL(22);

	// str[23] = b64t[(hash[4] >> 16) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 16), MM_SET1(63)));
	_B64TL(23);

	// str[24] = b64t[(hash[4] >> 10) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 10), MM_SET1(63)));
	_B64TL(24);

	// str[25] = b64t[(hash[4] >> 4) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 4), MM_SET1(63)));
	_B64TL(25);

#undef _B64TL

	for (i = 0; i < 4; i++){
		memcpy(str + 26 + 48 * i, str + 48 * i, 11);
	}
}

#endif	// USE_SHA1_SSE2	//////////////////////////////////////////////////


#ifdef USE_SHA1_AVX2	//////////////////////////////////////////////////

#undef MM_OR
#define MM_OR(a, b) _mm256_or_si256((a), (b))
#undef MM_AND
#define MM_AND(a, b) _mm256_and_si256((a), (b))
#undef MM_XOR
#define MM_XOR(a, b) _mm256_xor_si256((a), (b))
#undef MM_ADD
#define MM_ADD(a, b) _mm256_add_epi32((a), (b))

#undef MM_SLLI
#define MM_SLLI(a, b) _mm256_slli_epi32((a), (b))
#undef MM_SRLI
#define MM_SRLI(a, b) _mm256_srli_epi32((a), (b))

#undef MM_SET1
#define MM_SET1(a) _mm256_set1_epi32((a))

#undef MM_LOAD
#define MM_LOAD(a) _mm256_load_si256((__m256i *)(a))
#undef MM_STORE
#define MM_STORE(a, b) _mm256_store_si256((__m256i *)(a), (b))

#undef ROL32
#define ROL32(_val32, _nBits) (MM_OR(MM_SLLI((_val32), (_nBits)), MM_SRLI((_val32), 32-(_nBits))))

#undef SHABLK
#define SHABLK(t) (W[(t)&15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[((t)+13)&15], W[((t)+8)&15]), W[((t)+2)&15]), W[(t)&15]), 1))

#undef _RS0
#define _RS0(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_AND((w), MM_XOR((x), (y))), (y)), (i)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS00
#define _RS00(v,w,x,y,z) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_XOR(MM_AND((w), MM_XOR((x), (y))), (y)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _RS1
#define _RS1(v,w,x,y,z,i) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), (i)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R0
#define _R0(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_AND((w), MM_XOR((x), (y))), (y)), SHABLK(t)), MM_SET1(K0)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R1
#define _R1(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K1)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R2
#define _R2(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_OR(MM_AND(MM_OR((w), (x)), (y)), MM_AND((w), (x))), SHABLK(t)), MM_SET1(K2)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}

#undef _R3
#define _R3(v,w,x,y,z,t) { \
	z = MM_ADD((z), MM_ADD(MM_ADD(MM_ADD(MM_XOR(MM_XOR((w), (x)), (y)), SHABLK(t)), MM_SET1(K3)), ROL32(v,5))); \
	w = ROL32(w, 30); \
}


void sha1hash12byte_avx2(const char *input, __m256i *hash)
{
	__attribute__((aligned(32))) __m256i W[16];
	__attribute__((aligned(32))) __m256i a, b, c, d, e;
	__attribute__((aligned(32))) __m256i m_state[5];
	int i, j;
	char trip[13], tripkey[13];
	__attribute__((aligned(32))) uint32_t tmp0[8], tmp1[8], tmp2[8];

	const char trip64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/'
	};

	// SHA-1 initialization constants
	m_state[0] = MM_SET1(H0);
	m_state[1] = MM_SET1(H1);
	m_state[2] = MM_SET1(H2);
	m_state[3] = MM_SET1(H3);
	m_state[4] = MM_SET1(H4);

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 3; i++){
		W[i] = MM_LOAD(&input[32 * i]);
	}

	W[3] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(12 * 8);		// bits of Message Block (12 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS00(b, c, d, e, a);		// W[4] == 0
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4]==0, W[2])
	W[2] = ROL32(MM_XOR(W[15], W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4]==0)
	W[4] = ROL32(W[1], 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// trip test
	MM_STORE(tmp0, m_state[0]);

	for (i = 0; i < 8; i++){
		if (tmp0[i] >> 2 == trip_target_uint){
			MM_STORE(tmp1, m_state[1]);
			MM_STORE(tmp2, m_state[2]);

			trip[0] = trip64t[tmp0[i] >> 26];
			trip[1] = trip64t[(tmp0[i] >> 20) & 63];
			trip[2] = trip64t[(tmp0[i] >> 14) & 63];
			trip[3] = trip64t[(tmp0[i] >> 8) & 63];
			trip[4] = trip64t[(tmp0[i] >> 2) & 63];
			trip[5] = trip64t[(tmp0[i] << 4 | tmp1[i] >> 28) & 63];
			trip[6] = trip64t[(tmp1[i] >> 22) & 63];
			trip[7] = trip64t[(tmp1[i] >> 16) & 63];
			trip[8] = trip64t[(tmp1[i] >> 10) & 63];
			trip[9] = trip64t[(tmp1[i] >> 4) & 63];
			trip[10] = trip64t[(tmp1[i] << 2 | tmp2[i] >> 30) & 63];
			trip[11] = trip64t[(tmp2[i] >> 24) & 63];
			trip[12] = 0;

			if (! strncmp(opt_findtrip, trip, strlen(opt_findtrip))){
				for (j = 0; j < 3; j++){
					tripkey[0 + 4 * j] = input[3 + 32 * j + 4 * i];
					tripkey[1 + 4 * j] = input[2 + 32 * j + 4 * i];
					tripkey[2 + 4 * j] = input[1 + 32 * j + 4 * i];
					tripkey[3 + 4 * j] = input[0 + 32 * j + 4 * i];
				}
				tripkey[12] = 0;

				applog(LOG_INFO, "tripkey: #%s, trip: %s (yay!!!)", tripkey, trip);

				fprintf(fp_trip, "%s\t#%s\n", trip, tripkey);
				fflush(fp_trip);
			}
		}
	}

	for (i = 0; i < 5; i++){
		hash[i] = MM_XOR(hash[i], m_state[i]);
	}
}

void sha1hash80byte_2nd_avx2(const uint32_t *input, const uint32_t *prehash, char *str)
{
	__attribute__((aligned(32))) __m256i W[16];
	__attribute__((aligned(32))) __m256i a, b, c, d, e;
	__attribute__((aligned(32))) __m256i m_state[5];
	int i;
	__attribute__((aligned(32))) uint32_t tmp[8];

	const char b64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

	for (i = 0; i < 5; i++){
		m_state[i] = MM_SET1(prehash[i]);
	}

	a = m_state[0];
	b = m_state[1];
	c = m_state[2];
	d = m_state[3];
	e = m_state[4];

	for (i = 0; i < 4; i++){
		W[i] = MM_LOAD(&input[8 * i]);
	}

	W[4] = MM_SET1(0x80000000);		// padding

	W[15] = MM_SET1(80 * 8);		// bits of Message Block (80 bytes * 8 bits)

	// round 0 to 15
	_RS0(a, b, c, d, e, W[0]);
	_RS0(e, a, b, c, d, W[1]);
	_RS0(d, e, a, b, c, W[2]);
	_RS0(c, d, e, a, b, W[3]);
	_RS0(b, c, d, e, a, W[4]);
	_RS00(a, b, c, d, e);		// W[5] == 0
	_RS00(e, a, b, c, d);		// W[6] == 0
	_RS00(d, e, a, b, c);		// W[7] == 0
	_RS00(c, d, e, a, b);		// W[8] == 0
	_RS00(b, c, d, e, a);		// W[9] == 0
	_RS00(a, b, c, d, e);		// W[10] == 0
	_RS00(e, a, b, c, d);		// W[11] == 0
	_RS00(d, e, a, b, c);		// W[12] == 0
	_RS00(c, d, e, a, b);		// W[13] == 0
	_RS00(b, c, d, e, a);		// W[14] == 0
	_RS0(a, b, c, d, e, W[15]);

	// round 16 to 19
	// (t, W[t-3], W[t-8], W[t-14], W[t-16]) = (16, W[13]==0, W[8]==0, W[2], W[0])
	W[0] = ROL32(MM_XOR(W[2], W[0]), 1);
	_RS0(e, a, b, c, d, W[0]);

	// (17, W[14]==0, W[9]==0, W[3], W[1])
	W[1] = ROL32(MM_XOR(W[3], W[1]), 1);
	_RS0(d, e, a, b, c, W[1]);

	// (18, W[15], W[10]==0, W[4], W[2])
	W[2] = ROL32(MM_XOR(MM_XOR(W[15], W[4]), W[2]), 1);
	_RS0(c, d, e, a, b, W[2]);

	// (19, W[0], W[11]==0, W[5]==0, W[3])
	W[3] = ROL32(MM_XOR(W[0], W[3]), 1);
	_RS0(b, c, d, e, a, W[3]);

	// round 20 to 31
	// (20, W[1], W[12]==0, W[6]==0, W[4])
	W[4] = ROL32(MM_XOR(W[1], W[4]), 1);
	_RS1(a, b, c, d, e, W[4]);

	// (21, W[2], W[13]==0, W[7]==0, W[5]==0)
	W[5] = ROL32(W[2], 1);
	_RS1(e, a, b, c, d, W[5]);

	// (22, W[3], W[14]==0, W[8]==0, W[6]==0)
	W[6] = ROL32(W[3], 1);
	_RS1(d, e, a, b, c, W[6]);

	// (23, W[4], W[15], W[9]==0, W[7]==0)
	W[7] = ROL32(MM_XOR(W[4], W[15]), 1);
	_RS1(c, d, e, a, b, W[7]);

	// (24, W[5], W[0], W[10]==0, W[8]==0)
	W[8] = ROL32(MM_XOR(W[5], W[0]), 1);
	_RS1(b, c, d, e, a, W[8]);

	// (25, W[6], W[1], W[11]==0, W[9]==0)
	W[9] = ROL32(MM_XOR(W[6], W[1]), 1);
	_RS1(a, b, c, d, e, W[9]);

	// (26, W[7], W[2], W[12]==0, W[10]==0)
	W[10] = ROL32(MM_XOR(W[7], W[2]), 1);
	_RS1(e, a, b, c, d, W[10]);

	// (27, W[8], W[3], W[13]==0, W[11]==0)
	W[11] = ROL32(MM_XOR(W[8], W[3]), 1);
	_RS1(d, e, a, b, c, W[11]);

	// (28, W[9], W[4], W[14]==0, W[12]==0)
	W[12] = ROL32(MM_XOR(W[9], W[4]), 1);
	_RS1(c, d, e, a, b, W[12]);

	// (29, W[10], W[5], W[15], W[13]==0)
	W[13] = ROL32(MM_XOR(MM_XOR(W[10], W[5]), W[15]), 1);
	_RS1(b, c, d, e, a, W[13]);

	// (30, W[11], W[6], W[0], W[14]==0)
	W[14] = ROL32(MM_XOR(MM_XOR(W[11], W[6]), W[0]), 1);
	_RS1(a, b, c, d, e, W[14]);

	// (31, W[12], W[7], W[1], W[15])
	W[15] = ROL32(MM_XOR(MM_XOR(MM_XOR(W[12], W[7]), W[1]), W[15]), 1);
	_RS1(e, a, b, c, d, W[15]);

	// round 32 to 39
	_R1(d, e, a, b, c, 32);
	_R1(c, d, e, a, b, 33);
	_R1(b, c, d, e, a, 34);
	_R1(a, b, c, d, e, 35);
	_R1(e, a, b, c, d, 36);
	_R1(d, e, a, b, c, 37);
	_R1(c, d, e, a, b, 38);
	_R1(b, c, d, e, a, 39);

	// round 40 to 59
	_R2(a, b, c, d, e, 40);
	_R2(e, a, b, c, d, 41);
	_R2(d, e, a, b, c, 42);
	_R2(c, d, e, a, b, 43);
	_R2(b, c, d, e, a, 44);
	_R2(a, b, c, d, e, 45);
	_R2(e, a, b, c, d, 46);
	_R2(d, e, a, b, c, 47);
	_R2(c, d, e, a, b, 48);
	_R2(b, c, d, e, a, 49);
	_R2(a, b, c, d, e, 50);
	_R2(e, a, b, c, d, 51);
	_R2(d, e, a, b, c, 52);
	_R2(c, d, e, a, b, 53);
	_R2(b, c, d, e, a, 54);
	_R2(a, b, c, d, e, 55);
	_R2(e, a, b, c, d, 56);
	_R2(d, e, a, b, c, 57);
	_R2(c, d, e, a, b, 58);
	_R2(b, c, d, e, a, 59);

	// round 60 to 79
	_R3(a, b, c, d, e, 60);
	_R3(e, a, b, c, d, 61);
	_R3(d, e, a, b, c, 62);
	_R3(c, d, e, a, b, 63);
	_R3(b, c, d, e, a, 64);
	_R3(a, b, c, d, e, 65);
	_R3(e, a, b, c, d, 66);
	_R3(d, e, a, b, c, 67);
	_R3(c, d, e, a, b, 68);
	_R3(b, c, d, e, a, 69);
	_R3(a, b, c, d, e, 70);
	_R3(e, a, b, c, d, 71);
	_R3(d, e, a, b, c, 72);
	_R3(c, d, e, a, b, 73);
	_R3(b, c, d, e, a, 74);
	_R3(a, b, c, d, e, 75);
	_R3(e, a, b, c, d, 76);
	_R3(d, e, a, b, c, 77);
	_R3(c, d, e, a, b, 78);
	_R3(b, c, d, e, a, 79);

	// Add the working vars back into state
	m_state[0] = MM_ADD(m_state[0], a);
	m_state[1] = MM_ADD(m_state[1], b);
	m_state[2] = MM_ADD(m_state[2], c);
	m_state[3] = MM_ADD(m_state[3], d);
	m_state[4] = MM_ADD(m_state[4], e);

	// Base64 encode
#define _B64TL(i) { \
	str[(i)] = b64t[tmp[0]]; \
	str[(i)+48*1] = b64t[tmp[1]]; \
	str[(i)+48*2] = b64t[tmp[2]]; \
	str[(i)+48*3] = b64t[tmp[3]]; \
	str[(i)+48*4] = b64t[tmp[4]]; \
	str[(i)+48*5] = b64t[tmp[5]]; \
	str[(i)+48*6] = b64t[tmp[6]]; \
	str[(i)+48*7] = b64t[tmp[7]]; \
}

	// str[0] = b64t[hash[0] >> 26];
	MM_STORE(tmp, MM_SRLI(m_state[0], 26));
	_B64TL(0);

	// str[1] = b64t[(hash[0] >> 20) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 20), MM_SET1(63)));
	_B64TL(1);

	// str[2] = b64t[(hash[0] >> 14) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 14), MM_SET1(63)));
	_B64TL(2);

	// str[3] = b64t[(hash[0] >> 8) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 8), MM_SET1(63)));
	_B64TL(3);

	// str[4] = b64t[(hash[0] >> 2) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[0], 2), MM_SET1(63)));
	_B64TL(4);

	// str[5] = b64t[(hash[0] << 4 | hash[1] >> 28) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[0], 4), MM_SRLI(m_state[1], 28)), MM_SET1(63)));
	_B64TL(5);

	// str[6] = b64t[(hash[1] >> 22) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 22), MM_SET1(63)));
	_B64TL(6);

	// str[7] = b64t[(hash[1] >> 16) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 16), MM_SET1(63)));
	_B64TL(7);

	// str[8] = b64t[(hash[1] >> 10) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 10), MM_SET1(63)));
	_B64TL(8);

	// str[9] = b64t[(hash[1] >> 4) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[1], 4), MM_SET1(63)));
	_B64TL(9);

	// str[10] = b64t[(hash[1] << 2 | hash[2] >> 30) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[1], 2), MM_SRLI(m_state[2], 30)), MM_SET1(63)));
	_B64TL(10);

	// str[11] = b64t[(hash[2] >> 24) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 24), MM_SET1(63)));
	_B64TL(11);

	// str[12] = b64t[(hash[2] >> 18) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 18), MM_SET1(63)));
	_B64TL(12);

	// str[13] = b64t[(hash[2] >> 12) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 12), MM_SET1(63)));
	_B64TL(13);

	// str[14] = b64t[(hash[2] >> 6) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[2], 6), MM_SET1(63)));
	_B64TL(14);

	// str[15] = b64t[hash[2] & 63];
	MM_STORE(tmp, MM_AND(m_state[2], MM_SET1(63)));
	_B64TL(15);

	// str[16] = b64t[hash[3] >> 26];
	MM_STORE(tmp, MM_SRLI(m_state[3], 26));
	_B64TL(16);

	// str[17] = b64t[(hash[3] >> 20) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 20), MM_SET1(63)));
	_B64TL(17);

	// str[18] = b64t[(hash[3] >> 14) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 14), MM_SET1(63)));
	_B64TL(18);

	// str[19] = b64t[(hash[3] >> 8) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 8), MM_SET1(63)));
	_B64TL(19);

	// str[20] = b64t[(hash[3] >> 2) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[3], 2), MM_SET1(63)));
	_B64TL(20);

	// str[21] = b64t[(hash[3] << 4 | hash[4] >> 28) & 63];
	MM_STORE(tmp, MM_AND(MM_OR(MM_SLLI(m_state[3], 4), MM_SRLI(m_state[4], 28)), MM_SET1(63)));
	_B64TL(21);

	// str[22] = b64t[(hash[4] >> 22) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 22), MM_SET1(63)));
	_B64TL(22);

	// str[23] = b64t[(hash[4] >> 16) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 16), MM_SET1(63)));
	_B64TL(23);

	// str[24] = b64t[(hash[4] >> 10) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 10), MM_SET1(63)));
	_B64TL(24);

	// str[25] = b64t[(hash[4] >> 4) & 63];
	MM_STORE(tmp, MM_AND(MM_SRLI(m_state[4], 4), MM_SET1(63)));
	_B64TL(25);

#undef _B64TL

	for (i = 0; i < 8; i++){
		memcpy(str + 26 + 48 * i, str + 48 * i, 11);
	}
}

#endif	// USE_SHA1_AVX2	//////////////////////////////////////////////////


#ifdef __MINGW32__
/* 
 * public domain strtok_r() by Charlie Gordon
 *
 *   from comp.lang.c  9/14/2007
 *
 *      http://groups.google.com/group/comp.lang.c/msg/2ab1ecbb86646684
 *
 *     (Declaration that it's public domain):
 *      http://groups.google.com/group/comp.lang.c/msg/7c7b39328fefab9c
 */
char* strtok_r(
    char *str, 
    const char *delim, 
    char **nextp)
{
    char *ret;

    if (str == NULL)
    {
        str = *nextp;
    }

    str += strspn(str, delim);

    if (*str == '\0')
    {
        return NULL;
    }

    ret = str;

    str += strcspn(str, delim);

    if (*str)
    {
        *str++ = '\0';
    }

    *nextp = str;

    return ret;
}
#endif		// __MINGW32__


inline void encodeb64(const unsigned char* pch, char* buff)
{
  const char *pbase64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int mode = 0, left = 0;
  const int len = 20;
  const unsigned char *pchEnd = pch + len;
  while (pch < pchEnd) {
    int enc = *(pch++);
    if (mode == 0) {
      *buff++ = pbase64[enc >> 2];
      left = (enc & 3) << 4;
      mode = 1;
    }
    else if (mode == 1) {
      *buff++ = pbase64[left | (enc >> 4)];
      left = (enc & 15) << 2;
      mode = 2;
    }
    else {
      *buff++ = pbase64[left | (enc >> 6)];
      *buff++ = pbase64[enc & 63];
      mode = 0;
    }
  }
  *buff = pbase64[left];
//*(buff + 1) = 0;
}

uint32_t decodeb64chunk(const char* str)
{
  unsigned int dec = 0;
  const char *pbase64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (int i = 0; i < 4; i++)
  {
    char* pos = strchr(pbase64, str[i]);
    int v = (int)(pos - pbase64);
    dec = (dec << 6) | v;
  }
  dec <<= 8;
  uint32_t result = 0;
  be32enc(&result, dec); 
  return result;
}

inline void encodeb64chunk(const unsigned char* pch, char* buff)
{
  const char *pbase64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int mode = 0, left = 0;
  const int len = 3;
  const unsigned char *pchEnd = pch + len;
  while (pch < pchEnd) {
    int enc = *(pch++);
    if (mode == 0) {
      *buff++ = pbase64[enc >> 2];
      left = (enc & 3) << 4;
      mode = 1;
    }
    else if (mode == 1) {
      *buff++ = pbase64[left | (enc >> 4)];
      left = (enc & 15) << 2;
      mode = 2;
    }
    else {
      *buff++ = pbase64[left | (enc >> 6)];
      *buff++ = pbase64[enc & 63];
      mode = 0;
    }
  }
}

static unsigned short b64tbl1[0x10000];
static unsigned short b64tbl2[0x10000];
uint32_t searchchunks[128];
uint32_t searchchunks0;
uint32_t searchchunks1;
int nsearchchunks = 0;
char *findtrips[128];

void genb64tbl()
{
  unsigned char in[4] = "";
  unsigned char out[8] = "";
  for (int i = 0; i < 0x10000; i++) {
    in[0] = i & 0xff;
    in[1] = i >> 8;
    in[2] = 0;
    encodeb64chunk(in, out);
    b64tbl1[i] = out[0] | (out[1] << 8);
    in[0] = 0;
    in[1] = i & 0xff;
    in[2] = i >> 8;
    encodeb64chunk(in, out);
    b64tbl2[i] = out[2] | (out[3] << 8);
  }
  char *strbase = malloc(strlen(opt_findtrip) + 1);
  char *str = strbase;
  char *str2;
  char *saveptr = NULL;
  strcpy(str, opt_findtrip);
  for (int i = 0; str2 = strtok_r((char *)str, ",", &saveptr); i++) {
    findtrips[i] = str2;
    searchchunks[i] = decodeb64chunk(str2) & 0x00ffffff;
    if (i == 0) {
      searchchunks0 = ~searchchunks[i];
      searchchunks1 = searchchunks[i];
    }
    else {
      searchchunks0 &= ~searchchunks[i];
      searchchunks1 &= searchchunks[i];
    }
    nsearchchunks = i + 1;
    str = NULL;
  }
  if (searchchunks0 == 0) {
    for (int i = 0; i < nsearchchunks; i++) {
      searchchunks0 |= ~searchchunks[i];
    }
  }
  if (searchchunks1 == 0) {
    for (int i = 0; i < nsearchchunks; i++) {
      searchchunks1 |= searchchunks[i];
    }
  }
  // don't free strbase.
}

void encodeb64wide(const unsigned char* pch, unsigned short* buff)
{
  unsigned short sv;
  for (int i = 0; i < 7; i++) {
    sv = pch[0] | (pch[1] << 8);
    *buff++ = b64tbl1[sv];
    sv = pch[1] | (pch[2] << 8);
    *buff++ = b64tbl2[sv];
    pch += 3;
  }
}

void tbltest()
{
  unsigned char in[21] = "12345678901234567890";
  char out[30] = "";
  genb64tbl();
  encodeb64(in, out);
  printf("expected: %s\n", out);
  encodeb64wide(in, (unsigned short *)out);
  printf("actual  : %s\n", out);
}

uint32_t sha1coinhash(void *state, const void *input)
{
  char str[38] __attribute__((aligned(32))); // 26 + 11 + 1
  char trip[28] __attribute__((aligned(32))); // 26 + 1 + padding
  char tripkey[13] = "";
  uint32_t prehash[5] __attribute__((aligned(32)));
  uint32_t hash[5] __attribute__((aligned(32))) = { 0 };
  uint32_t prehash0;
  uint32_t hash4 = 0;

#ifdef USE_SHA1_OPENSSL
  SHA1(input, 20 * 4, (void *)prehash);
#else
	sha1hash80byte(input, prehash);
#endif

#if 0
  encodeb64((const unsigned char *)prehash, (unsigned char *)str);
#else
  encodeb64wide((const unsigned char *)prehash, (unsigned short *)str);
#endif
  memcpy(&str[26], str, 11);
//str[37] = 0;
  for (int i = 0; i < 26; i++) {
#ifdef USE_SHA1_OPENSSL
    SHA1((const unsigned char*)&str[i], 12, (unsigned char *)prehash);
#else
  	sha1hash12byte(str + i, prehash);
#endif
#define TRIP
#if defined(TRIP)
    prehash0 = prehash[0] & 0x00ffffff;
    if ((prehash0 & searchchunks1) && (~prehash0 & searchchunks0)) {
      for (int j = 0; j < nsearchchunks; j++) {
        if (prehash0 == searchchunks[j]) {
          encodeb64wide((const unsigned char *)prehash, (unsigned short *)trip);
          memcpy(tripkey, &str[i], 12);
          trip[12] = 0;
          int triplen = strlen(findtrips[j]);
          int result = !memcmp(trip, findtrips[j], triplen);
          if (result) {
            applog(LOG_INFO, "tripkey: #%s, trip: %s %s", tripkey, trip, "(yay!!!)");
          }
        }
      }
    }
#endif
#define CHEAT
#if !defined(CHEAT)
    hash[0] ^= prehash[0];
    hash[1] ^= prehash[1];
    hash[2] ^= prehash[2];
    hash[3] ^= prehash[3];
#endif
    //hash[4] ^= prehash[4];
    hash4 ^= prehash[4];
  }
#if !defined(CHEAT)
  memcpy(state, hash, 20);
#else
  //memcpy((char *)state + 16, &hash[4], 4);
  //*(uint32_t *)state = hash4;
#endif
  return hash4;
}


#ifdef USE_SHA1_PIQPU
static inline int scanhash_sha1coin_qpu(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce - 1;
	uint32_t ntmp;

	uint32_t data[4] __attribute__((aligned(32)));

	char str[38] __attribute__((aligned(32))) = {0};	// 26 + 11 + 1
	char tripkey[38] __attribute__((aligned(32))) = {0};

	uint32_t prehash[5] __attribute__((aligned(32)));
	uint32_t hash[8] __attribute__((aligned(32)));
	uint32_t m_state[5] __attribute__((aligned(32)));
	
	int i, j, k, l;
	int tes;
    unsigned int strw[10];
    unsigned int wtmp[NUM_QPUS][16][10];
    char strtest[38] __attribute__((aligned(32))); // 26 + 11 + 1

	// process 1st block of SHA-1, 512bits, 64bytes
	sha1hash80byte_1st(pdata, prehash);

	// setup data for 2nd block of SHA-1, 16bytes
	memcpy(data, pdata + 16, 16);

	qpu_ini();

	do {
		ntmp=n;
		data[3] = ++ntmp;

		// QPU Data Set
    	for (i=0; i < NUM_QPUS; i++) {
    	    arm_map->uniforms[i][4] = prehash[0];
    	    arm_map->uniforms[i][5] = prehash[1];
    	    arm_map->uniforms[i][6] = prehash[2];
    	    arm_map->uniforms[i][7] = prehash[3];
    	    arm_map->uniforms[i][8] = prehash[4];
    	    arm_map->uniforms[i][9] = data[0];
    	    arm_map->uniforms[i][10] = data[1];
    	    arm_map->uniforms[i][11] = data[2];
    	    arm_map->uniforms[i][12] = data[3];
    	}

    	unsigned ret = execute_qpu(mb, NUM_QPUS, vc_msg, 1, 10000);

		// QPU Outputdata Read & Check
		for(i=0 ; i<NUM_QPUS ; i++){
			for(l=0; l<QPUTHR; l++){
				// hash check
				if (arm_map->results[i][64+l] <= ptarget[7]){
					memset(hash, 0, 12);
					for(j=0; j<4; j++){
						hash[j+3] = swab32(arm_map->results[i][(j*16)+l]);
					}
					hash[7] = arm_map->results[i][(64)+l];
					data[3] = ntmp + (i * QPUTHR) + l;
					if (fulltest(hash, ptarget)){
						pdata[19] = data[3];
						*hashes_done = n - first_nonce + 1;
						qpu_end();
						return 1;
					}
				}
			}
		}
		n += NUM_QPUS * QPUTHR;
	} while (n < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	qpu_end();

	return 0;
}

#endif

#ifdef USE_SHA1_OPT
static inline int scanhash_sha1coin_opt(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce - 1;

	uint32_t data[4] __attribute__((aligned(32)));

	char str[38] __attribute__((aligned(32))) = {0};	// 26 + 11 + 1
	char tripkey[12] __attribute__((aligned(32))) = {0};

	uint32_t prehash[5] __attribute__((aligned(32)));
	uint32_t hash[8] __attribute__((aligned(32)));
	int i, j, k;

	// process 1st block of SHA-1, 512bits, 64bytes
	sha1hash80byte_1st(pdata, prehash);

	// setup data for 2nd block of SHA-1, 16bytes
	memcpy(data, pdata + 16, 16);

	do {
		data[3] = ++n;

		// process 2nd block of SHA-1, 16bytes, generate str
		sha1hash80byte_2nd_opt(data, prehash, str);

		memset(hash, 0, 32);

		for (k = 0; k < 26; k++){
			// generate tripkey from str
			memcpy(tripkey, str + k, 12);

			// compute hash and search trip
			sha1hash12byte_opt(tripkey, hash + 3);
		}

		hash[7] = swab32(hash[7]);

		if (hash[7] <= ptarget[7]){
			for (i = 0; i < 7; i++){
				hash[i] = swab32(hash[i]);
			}

			if (fulltest(hash, ptarget)){
				pdata[19] = data[3];
				*hashes_done = n - first_nonce + 1;
				return 1;
			}
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}

#endif	// USE_SHA1_OPT


#ifdef USE_SHA1_NEON
static inline int scanhash_sha1coin_neon(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce - 1;

	uint32_t data[4 * 4] __attribute__((aligned(32)));

	char str[4 * 48] __attribute__((aligned(32))) = {0};	// (26 + 11 + 1 + padding) * 4
	char tripkey[4 * 4 * 3] __attribute__((aligned(32)));

	uint32_t prehash[5] __attribute__((aligned(32)));
	uint32_t hash[8] __attribute__((aligned(32))) = {0};
	uint32x4_t hash_x4[5] __attribute__((aligned(32)));
	int i, j, k;
	__attribute__((aligned(16))) uint32_t tmp0[4], tmp1[4], tmp2[4], tmp3[4], tmp4[4];

	// process 1st block of SHA-1, 512bits, 64bytes
	sha1hash80byte_1st(pdata, prehash);

	// setup data for 2nd block of SHA-1, 16bytes
	memcpy(data, pdata + 16, 16);
	for (i = 3; i >= 0; i--){
		for (j = 0; j < 4; j++){
			data[4 * i + j] = data[i];
		}
	}

	do {
		for (i = 0; i < 4; i++){
			data[4 * 3 + i] = ++n;
		}

		// process 2nd block of SHA-1, 16bytes, generate str
		sha1hash80byte_2nd_neon(data, prehash, str);

		for (i = 0; i < 5; i++){
			hash_x4[i] = vdupq_n_u32(0);
		}

		for (k = 0; k < 26; k++){
			// generate tripkey table from str
			for (i = 0; i < 4; i++){
				for (j = 0; j < 3; j++){
					tripkey[3 + 4 * i + 16 * j] = str[k + 0 + 48 * i + 4 * j];
					tripkey[2 + 4 * i + 16 * j] = str[k + 1 + 48 * i + 4 * j];
					tripkey[1 + 4 * i + 16 * j] = str[k + 2 + 48 * i + 4 * j];
					tripkey[0 + 4 * i + 16 * j] = str[k + 3 + 48 * i + 4 * j];
				}
			}

			// compute hash and search trip
			sha1hash12byte_neon(tripkey, hash_x4);
		}

		*(uint32x4_t *)tmp4 = hash_x4[4];

		for (i = 0; i < 4; i++){
			hash[7] = swab32(tmp4[i]);

			if (hash[7] <= ptarget[7]){
				*(uint32x4_t *)tmp0 = hash_x4[0];
				*(uint32x4_t *)tmp1 = hash_x4[1];
				*(uint32x4_t *)tmp2 = hash_x4[2];
				*(uint32x4_t *)tmp3 = hash_x4[3];

				hash[3] = swab32(tmp0[i]);
				hash[4] = swab32(tmp1[i]);
				hash[5] = swab32(tmp2[i]);
				hash[6] = swab32(tmp3[i]);

				if (fulltest(hash, ptarget)){
					pdata[19] = data[4 * 3 + i];
					*hashes_done = n - first_nonce + 1;
					return 1;
				}
			}
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}

#endif	// USE_SHA1_NEON


#ifdef USE_SHA1_SSE2
static inline int scanhash_sha1coin_sse2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce - 1;

	uint32_t data[4 * 4] __attribute__((aligned(32)));

	char str[4 * 48] __attribute__((aligned(32))) = {0};	// (26 + 11 + 1 + padding) * 4
	char tripkey[4 * 4 * 3] __attribute__((aligned(32)));

	uint32_t prehash[5] __attribute__((aligned(32)));
	uint32_t hash[8] __attribute__((aligned(32))) = {0};
	__m128i hash_m128i[5] __attribute__((aligned(32)));
	int i, j, k;
	__attribute__((aligned(16))) uint32_t tmp0[4], tmp1[4], tmp2[4], tmp3[4], tmp4[4];

	// process 1st block of SHA-1, 512bits, 64bytes
	sha1hash80byte_1st(pdata, prehash);

	// setup data for 2nd block of SHA-1, 16bytes
	memcpy(data, pdata + 16, 16);
	for (i = 3; i >= 0; i--){
		for (j = 0; j < 4; j++){
			data[4 * i + j] = data[i];
		}
	}

	do {
		for (i = 0; i < 4; i++){
			data[4 * 3 + i] = ++n;
		}

		// process 2nd block of SHA-1, 16bytes, generate str
		sha1hash80byte_2nd_sse2(data, prehash, str);

		for (i = 0; i < 5; i++){
			hash_m128i[i] = _mm_set1_epi32(0);
		}

		for (k = 0; k < 26; k++){
			// generate tripkey table from str
			for (i = 0; i < 4; i++){
				for (j = 0; j < 3; j++){
					tripkey[3 + 4 * i + 16 * j] = str[k + 0 + 48 * i + 4 * j];
					tripkey[2 + 4 * i + 16 * j] = str[k + 1 + 48 * i + 4 * j];
					tripkey[1 + 4 * i + 16 * j] = str[k + 2 + 48 * i + 4 * j];
					tripkey[0 + 4 * i + 16 * j] = str[k + 3 + 48 * i + 4 * j];
				}
			}

			// compute hash and search trip
			sha1hash12byte_sse2(tripkey, hash_m128i);
		}

		_mm_store_si128((__m128i *)tmp4, hash_m128i[4]);

		for (i = 0; i < 4; i++){
			hash[7] = swab32(tmp4[i]);

			if (hash[7] <= ptarget[7]){
				_mm_store_si128((__m128i *)tmp0, hash_m128i[0]);
				_mm_store_si128((__m128i *)tmp1, hash_m128i[1]);
				_mm_store_si128((__m128i *)tmp2, hash_m128i[2]);
				_mm_store_si128((__m128i *)tmp3, hash_m128i[3]);

				hash[3] = swab32(tmp0[i]);
				hash[4] = swab32(tmp1[i]);
				hash[5] = swab32(tmp2[i]);
				hash[6] = swab32(tmp3[i]);

				if (fulltest(hash, ptarget)){
					pdata[19] = data[4 * 3 + i];
					*hashes_done = n - first_nonce + 1;
					return 1;
				}
			}
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}

#endif	// USE_SHA1_SSE2


#ifdef USE_SHA1_AVX2
static inline int scanhash_sha1coin_avx2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce - 1;

	uint32_t data[8 * 4] __attribute__((aligned(32)));

	char str[8 * 48] __attribute__((aligned(32))) = {0};	// (26 + 11 + 1 + padding) * 8
	char tripkey[8 * 4 * 3] __attribute__((aligned(32)));

	uint32_t prehash[5] __attribute__((aligned(32)));
	uint32_t hash[8] __attribute__((aligned(32))) = {0};
	__m256i hash_m256i[5] __attribute__((aligned(32)));
	int i, j, k;
	__attribute__((aligned(32))) uint32_t tmp0[8], tmp1[8], tmp2[8], tmp3[8], tmp4[8];

	// process 1st block of SHA-1, 512bits, 64bytes
	sha1hash80byte_1st(pdata, prehash);

	// setup data for 2nd block of SHA-1, 16bytes
	memcpy(data, pdata + 16, 16);
	for (i = 3; i >= 0; i--){
		for (j = 0; j < 8; j++){
			data[8 * i + j] = data[i];
		}
	}

	do {
		for (i = 0; i < 8; i++){
			data[8 * 3 + i] = ++n;
		}

		// process 2nd block of SHA-1, 16bytes, generate str
		sha1hash80byte_2nd_avx2(data, prehash, str);

		for (i = 0; i < 5; i++){
			hash_m256i[i] = _mm256_set1_epi32(0);
		}

		for (k = 0; k < 26; k++){
			// generate tripkey table from str
			for (i = 0; i < 8; i++){
				for (j = 0; j < 3; j++){
					tripkey[3 + 4 * i + 32 * j] = str[k + 0 + 48 * i + 4 * j];
					tripkey[2 + 4 * i + 32 * j] = str[k + 1 + 48 * i + 4 * j];
					tripkey[1 + 4 * i + 32 * j] = str[k + 2 + 48 * i + 4 * j];
					tripkey[0 + 4 * i + 32 * j] = str[k + 3 + 48 * i + 4 * j];
				}
			}

			// compute hash and search trip
			sha1hash12byte_avx2(tripkey, hash_m256i);
		}

		_mm256_store_si256((__m256i *)tmp4, hash_m256i[4]);

		for (i = 0; i < 8; i++){
			hash[7] = swab32(tmp4[i]);

			if (hash[7] <= ptarget[7]){
				_mm256_store_si256((__m256i *)tmp0, hash_m256i[0]);
				_mm256_store_si256((__m256i *)tmp1, hash_m256i[1]);
				_mm256_store_si256((__m256i *)tmp2, hash_m256i[2]);
				_mm256_store_si256((__m256i *)tmp3, hash_m256i[3]);

				hash[3] = swab32(tmp0[i]);
				hash[4] = swab32(tmp1[i]);
				hash[5] = swab32(tmp2[i]);
				hash[6] = swab32(tmp3[i]);

				if (fulltest(hash, ptarget)){
					pdata[19] = data[8 * 3 + i];
					*hashes_done = n - first_nonce + 1;
					return 1;
				}
			}
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}

#endif	// USE_SHA1_AVX2


int scanhash_sha1coin(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
    uint32_t max_nonce, unsigned long *hashes_done)
{
  const uint32_t first_nonce = pdata[19];
  uint32_t n = first_nonce - 1;
  uint32_t endiandata[32];
  uint32_t hash[8] __attribute__((aligned(32)));
#if defined(CHEAT)
  uint32_t hash7;
#endif

#ifdef USE_SHA1_AVX2
	return scanhash_sha1coin_avx2(thr_id, pdata, ptarget, max_nonce, hashes_done);
#endif

#ifdef USE_SHA1_SSE2
	return scanhash_sha1coin_sse2(thr_id, pdata, ptarget, max_nonce, hashes_done);
#endif

#ifdef USE_SHA1_NEON
	return scanhash_sha1coin_neon(thr_id, pdata, ptarget, max_nonce, hashes_done);
#endif

#ifdef USE_SHA1_PIQPU
	return scanhash_sha1coin_qpu(thr_id, pdata, ptarget, max_nonce, hashes_done);
#endif

#ifdef USE_SHA1_OPT
	return scanhash_sha1coin_opt(thr_id, pdata, ptarget, max_nonce, hashes_done);
#endif

  hash[0] = 0;
  hash[1] = 0;
  hash[2] = 0;
  for (int kk = 0; kk < 32; kk++) {
    be32enc(&endiandata[kk], ((uint32_t*)pdata)[kk]);
  }
  do {
    pdata[19] = ++n;
    be32enc(&endiandata[19], n); 
#if defined(CHEAT)
    hash7 = sha1coinhash(NULL, endiandata);
    if (!(hash7 & 0xfffffc00)) {
      hash[7] = hash7;
      if (fulltest(hash, ptarget)) {
        *hashes_done = n - first_nonce + 1;
        return 1;
      }
    }
#else
    sha1coinhash(&hash[3], endiandata);
    if (!(hash[7] & 0xfffffc00) && fulltest(hash, ptarget)) {
      *hashes_done = n - first_nonce + 1;
      return 1;
    }
#endif
  } while (n < max_nonce && !work_restart[thr_id].restart);
  *hashes_done = n - first_nonce + 1;
  pdata[19] = n;
  return 0;
}


#ifdef USE_SHA1_PIQPU
int qpu_ini(){
	unsigned vc_uniforms;
	unsigned vc_code;
	unsigned vc_input;
	unsigned vc_results;
	unsigned ptr;

	const char w64t[] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
	};

    mb = mbox_open();
    if (qpu_enable(mb, 1)){
        fprintf(stderr, "QPU enable failed.\n");
        return -1;
    }
    
    size = 1024 * 1024;
    handle = mem_alloc(mb, size, 4096, GPU_MEM_FLG);
    if (!handle){
        fprintf(stderr, "Unable to allocate %d bytes of GPU memory", size);
        return -2;
    }
    
    ptr = mem_lock(mb, handle);									// GPUに渡すmemoryのpointer
    arm_ptr = mapmem(ptr + GPU_MEM_MAP, size);

    arm_map = (struct memory_map *)arm_ptr;
    memset(arm_map, 0x0, sizeof(struct memory_map));
    vc_input = ptr + offsetof(struct memory_map, input);
    vc_results = ptr + offsetof(struct memory_map, results);
    vc_uniforms = ptr + offsetof(struct memory_map, uniforms);	
    vc_code = ptr + offsetof(struct memory_map, code);
    vc_msg = ptr + offsetof(struct memory_map, msg);
    memcpy(arm_map->code, qpu_code, sizeof(qpu_code));

    for (int i=0; i < NUM_QPUS; i++){
        arm_map->uniforms[i][0] = vc_input + i * sizeof(unsigned) * VPM_SIZE;
        arm_map->uniforms[i][1] = vc_results + i * sizeof(unsigned) * VPM_SIZE;
        arm_map->uniforms[i][2] = i;
        arm_map->uniforms[i][3] = NUM_QPUS;
        arm_map->msg[i][0] = vc_uniforms + i * sizeof(unsigned) * UNIFORMS;
        arm_map->msg[i][1] = vc_code;
		for (int j=0; j < 64; j++){
			arm_map->input[i][j] = (unsigned int)w64t[j];
		}
    }
}

void qpu_end()
{
	unmapmem(arm_ptr, size);
	mem_unlock(mb, handle);
	mem_free(mb, handle);
	qpu_enable(mb, 0);
}

#endif
