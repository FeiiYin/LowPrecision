/**
 * @infi
 * float16 (unsigned short) and float transform function
 * float16 add & sub
 * float16 multiply & divide
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

#define _XINLINE_ inline
#define TEST 0

/*
 * Half-precision constants
 */

#define NPY_HALF_ZERO   (0x0000u)
#define NPY_HALF_PZERO  (0x0000u)
#define NPY_HALF_NZERO  (0x8000u)
#define NPY_HALF_ONE    (0x3c00u)
#define NPY_HALF_NEGONE (0xbc00u)
#define NPY_HALF_PINF   (0x7c00u)
#define NPY_HALF_NINF   (0xfc00u)
#define NPY_HALF_NAN    (0x7e00u)

#define NPY_MAX_HALF    (0x7bffu)

/*
 * Bit-level conversions
 */
unsigned short FloatbitsToHalfbits(float ff)
{
    unsigned int f = *((unsigned int*)&ff);
    unsigned int f_exp, f_sig;
    unsigned short h_sgn, h_exp, h_sig;

    h_sgn = (unsigned short) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                unsigned short ret = (unsigned short) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (unsigned short) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
            return (unsigned short) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (f_exp < 0x33000000u) {
            /* If f != 0, it underflowed to 0 */
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));

        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */

        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }

        h_sig = (unsigned short) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (unsigned short) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (unsigned short) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);

    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }

    h_sig = (unsigned short) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    h_sig += h_exp;
    return h_sgn + h_sig;
}

float HalfbitsToFloatbits(unsigned short h)
{
    unsigned short h_exp, h_sig;
    unsigned int f_sgn, f_exp, f_sig;
    unsigned int ret = 0;

    h_exp = (h&0x7c00u);
    f_sgn = ((unsigned int)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((unsigned int)(127 - 15 - h_exp)) << 23;
            f_sig = ((unsigned int)(h_sig&0x03ffu)) << 13;
            ret = f_sgn + f_exp + f_sig;
            break;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            ret = f_sgn + 0x7f800000u + (((unsigned int)(h&0x03ffu)) << 13);
            break;
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            ret = f_sgn + (((unsigned int)(h&0x7fffu) + 0x1c000u) << 13);
            break;
    }
    return *((float*)&ret);
}

/* float16 -> float */
_XINLINE_ float Float16ToFloat(unsigned short h)
{
    float f = float(((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13));
    float ret = HalfbitsToFloatbits(h);
#if TEST
    std::cout << std::bitset<32>(*((unsigned int*)&ret)) << std::endl;
#endif
    return ret;
}

/* float -> float16 */
_XINLINE_ unsigned short FloatToFloat16(float f)
{
    unsigned short ret = FloatbitsToHalfbits(f);
#if TEST
    std::cout << std::bitset<16>(*((unsigned short*)&ret)) << std::endl;
#endif
    return ret;
}


void test (float a) {

    cout<< "origin:" << a << endl;
    std::cout << std::bitset<32>(*((unsigned int*)&a)) << std::endl;
    unsigned short b = FloatToFloat16(a);
    a = Float16ToFloat(b);
    cout << a << endl;
}


const unsigned short FLOAT16_INF = 0x7c00u;
const unsigned short FLOAT16_NEGATIVE_INF = 0xfc00u;
const unsigned short FLOAT16_ZERO = 0x0u;
const unsigned short FLOAT16_NEGATIVE_ZERO = 0x8000u;
/* complement - bu ma */
inline unsigned short cal_complement (unsigned short sig, unsigned short tal) {
    /* 00.10000000000 */
    if (! sig) return tal;
//    unsigned short lowbit = tal & (-tal);
//    (tal ^= (0x1fffu));
//    tal ^= (lowbit - 1);
//    tal += lowbit;
    (tal ^= (0x1fffu));
    tal++;
    return tal;
}

unsigned short Float16Add(unsigned short a, unsigned short b)
{
    if (a == FLOAT16_INF || b == FLOAT16_INF)
        return FLOAT16_INF;
    if (a == FLOAT16_NEGATIVE_INF || b == FLOAT16_NEGATIVE_INF)
        return FLOAT16_NEGATIVE_INF;
    if (a == FLOAT16_ZERO || a == FLOAT16_NEGATIVE_ZERO)
        return b;
    if (b == FLOAT16_ZERO || b == FLOAT16_NEGATIVE_ZERO)
        return a;

//    const unsigned short TAIL_LENGTH = 10;
    unsigned short a_sig = (a&0x8000u) >> 15, a_exp = (a&0x7c00u) >> 10, a_tal = (a&0x03ffu)|(0x0400u);
    unsigned short b_sig = (b&0x8000u) >> 15, b_exp = (b&0x7c00u) >> 10, b_tal = (b&0x03ffu)|(0x0400u);

    unsigned short c_sig = 0, c_exp = 0, c_tal = 0;

    /* exp differ */
    unsigned short exp_dif;
    if (a_exp > b_exp) {
        exp_dif = a_exp - b_exp;
        b_tal >>= exp_dif;
        c_exp = a_exp;
    }
    else {
        exp_dif = b_exp - a_exp;
        a_tal >>= exp_dif;
        c_exp = b_exp;
    }

    a_tal = cal_complement(a_sig, a_tal);
    b_tal = cal_complement(b_sig, b_tal);
    c_tal = a_tal + b_tal;

    if ((c_tal & 0x0fffu) == 0) {
        c_sig = 0;
        c_exp = 0;
        c_tal = 0;
    }
    else if ((c_tal & 0x1800u) == 0x0000u || (c_tal & 0x1800u) == 0x1800u) {
        if ((c_tal & 0x1800u) == 0x0000u)
            c_sig = 0;
        else
            c_sig = 1;
        c_tal &= 0x07ffu;
        unsigned short temp = c_sig << 10;
        /* left standard */
        while ((c_tal & 0x0400u) == temp) {
            c_tal <<= 1;
            c_exp--;
            if (c_exp == 0) break;
        }
        c_tal &= 0x03ffu;
        if (! c_tal)
            c_exp++;
    }
    else if ((c_tal & 0x1800u) == 0x1000u) {
        // 10->right into 11
        c_sig = 1;
        c_tal >>= 1;
        c_exp++;
        c_tal &= 0x03ffu;
    }
    else if ((c_tal & 0x1800u) == 0x0800u) {
        c_sig = 0;
        c_tal >>= 1;
        c_exp++;
        c_tal &= 0x03ffu;
    }
    /* amazing funds negative number need to invert */
    /* lowbit might be faster than directly invert and plus 1 */
    if (c_sig == 1 && c_tal) {
        unsigned short lowbit = c_tal & (-c_tal);
        c_tal ^= 0x03ffu;
        c_tal ^= (lowbit - 1);
        c_tal += lowbit;
    }

#if TEST
    unsigned short ret = (c_sig << 15) + (c_exp << 10) + c_tal;
    std::cout << std::bitset<16>(*((unsigned short*)&ret)) << std::endl;
#endif
    /* check the overflow & underflow */
    if (c_exp == 0x1fu)
        c_tal = 0;
    return (c_sig << 15) + (c_exp << 10) + c_tal;
}

unsigned short Float16Sub(unsigned short a, unsigned short b)
{
    /* invert the sign */
    b ^= 0x8000u;
    return Float16Add(a, b);
}

const unsigned short FLOAT16_SHIFT_BASE = 0xfu;

unsigned short Float16Mul(unsigned short a, unsigned short b) {
    /* check zero and inf */
    if (a == FLOAT16_ZERO || a == FLOAT16_NEGATIVE_ZERO)
        return FLOAT16_ZERO;
    if (b == FLOAT16_ZERO || b == FLOAT16_NEGATIVE_ZERO)
        return FLOAT16_ZERO;
    if (a == FLOAT16_INF || b == FLOAT16_INF)
        return FLOAT16_INF;
    if (a == FLOAT16_NEGATIVE_INF || b == FLOAT16_NEGATIVE_INF)
        return FLOAT16_NEGATIVE_INF;
    /* achieve basic value */
    unsigned short a_sig = (a&0x8000u) >> 15, a_exp = (a&0x7c00u) >> 10, a_tal = (a&0x03ffu)|(0x0400u);
    unsigned short b_sig = (b&0x8000u) >> 15, b_exp = (b&0x7c00u) >> 10, b_tal = (b&0x03ffu)|(0x0400u);

    a_exp++;
    b_exp++;

    unsigned short c_sig, c_exp, c_tal;
    /* A*B = (Ta*Tb)*2^(Sa+Sb) */
    /* calculate the shift code */
    b_exp ^= 0x10u;
    b_exp |= (b_exp & 0x10u) << 1;
    c_exp = a_exp + b_exp;

    if (c_exp & 0x20u) {
        if (c_exp & 0x10u)
            return FLOAT16_NEGATIVE_INF;
        else
            return FLOAT16_INF;
    }
    c_exp &= 0x1fu;

    /* calculate the tail code */
    c_tal = 0;
    for (unsigned short i = 10; i >= (unsigned short) 6; --i) {
        if (b_tal & (1<<i)) {
            c_tal += a_tal;
        }
        a_tal >>= 1;
    }
    /* at most process one bit */
    if (c_tal & 0x800u) {
        c_tal >>= 1;
    } else {
        c_exp--;
    }

    c_tal &= 0x3ffu;

    /* calculate the sign */
    c_sig = a_sig ^ b_sig;
#if TEST
    unsigned short ret = (c_sig << 15) + (c_exp << 10) + c_tal;
    std::cout << "multi : " << std::bitset<16>(*((unsigned short*)&ret)) << std::endl;
#endif
    return (c_sig << 15) + (c_exp << 10) + c_tal;
}

unsigned short Float16Div(unsigned short a, unsigned short b) {
    /* check zero and inf */
    if (b == FLOAT16_ZERO || b == FLOAT16_NEGATIVE_ZERO)
        return FLOAT16_INF;
    if (a == FLOAT16_ZERO || a == FLOAT16_NEGATIVE_ZERO)
        return FLOAT16_ZERO;
    if (a == FLOAT16_INF || a == FLOAT16_NEGATIVE_INF)
        return FLOAT16_INF;
    if (b == FLOAT16_NEGATIVE_INF || b == FLOAT16_NEGATIVE_INF)
        return FLOAT16_ZERO;
    /* achieve basic value */
    unsigned short a_sig = (a&0x8000u) >> 15, a_exp = (a&0x7c00u) >> 10, a_tal = (a&0x03ffu)|(0x0400u);
    unsigned short b_sig = (b&0x8000u) >> 15, b_exp = (b&0x7c00u) >> 10, b_tal = (b&0x03ffu)|(0x0400u);

    a_exp++;
    b_exp++;

    unsigned short c_sig, c_exp, c_tal;
    /* A*B = (Ta*Tb)*2^(Sa+Sb) */
    /* calculate the shift code */

    b_exp |= (b_exp & 0x10u) << 1;
    b_exp ^= 0xfu;
    b_exp++;
    c_exp = a_exp + b_exp;

    if (c_exp & 0x20u) {
        if (c_exp & 0x10u)
            return FLOAT16_NEGATIVE_INF;
        else
            return FLOAT16_INF;
    }
    c_exp &= 0x1fu;
    c_exp--;

    /* calculate the tail code */
    c_tal = 0;
    unsigned short low_bit = b_tal & (-b_tal);

    for (unsigned short i = low_bit; i > 0; i >>= 1) {
        if (a_tal >= b_tal) {
            c_tal += i;
            a_tal -= b_tal;
        }
        b_tal >>= 1;
    }

    while (!(c_tal & low_bit)) {
        c_tal <<= 1;
        c_exp--;
    }

    while (!(low_bit & 0x400u)) {
        low_bit <<= 1;
        c_tal <<= 1;
    }
    c_tal &= 0x3ffu;

    /* calculate the sign */
    c_sig = a_sig ^ b_sig;
#if TEST
    unsigned short ret = (c_sig << 15) + (c_exp << 10) + c_tal;
    std::cout << "divide : " << std::bitset<16>(*((unsigned short*)&ret)) << std::endl;
#endif
    return (c_sig << 15) + (c_exp << 10) + c_tal;
}

int main()
{
    float a = 20, b = -101;
    /* calculate +*-/ test */
    while (true)
    {
        scanf("%f %f", &a, &b);
        unsigned short aa = FloatToFloat16(a), bb = FloatToFloat16(b);
        unsigned short cc = Float16Mul(aa, bb), dd = Float16Div(aa, bb);
        float c = Float16ToFloat(cc), d = Float16ToFloat(dd);

        unsigned short ee = Float16Add(aa, bb), ff = Float16Sub(aa, bb);
        float e = Float16ToFloat(ee), f = Float16ToFloat(ff);
        cout << "a + b = " << e << endl;
        cout << "a - b = " << f << endl;
        cout << "a * b = " << c << endl;
        cout << "a / b = " << d << endl;
    }

    /* calculate +/- test */
//    while (true)
//    {
//        scanf("%f %f", &a, &b);
//        unsigned short aa = FloatToFloat16(a), bb = FloatToFloat16(b);
//        unsigned short cc = Float16Add(aa, bb), dd = Float16Sub(aa, bb);
//        float c = Float16ToFloat(cc), d = Float16ToFloat(dd);
//        cout << c << " " << d << endl;
//    }

    /* transform test */
//    float a = 111111111111;
//    test(a);
//    a = 0.25;
//    test(a);
//    a = 0;
//    test(a);
//    a = -111111111111111;
//    test(a);
    return 0;
}

