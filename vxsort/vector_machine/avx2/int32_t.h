template <>
class vxsort_machine_traits<int32_t, AVX2> {
   public:
    typedef int32_t T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef int32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return false; }
    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < N);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + N * remainder)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi32((int32_t *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi32((int32_t *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE TV broadcast(int32_t pivot) { return _mm256_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(a, b))); }

    static TV shift_right(TV v, int i) { return _mm256_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm256_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi32(a, b); };

    static INLINE TV pack_ordered(TV , TV ) { TV tmp = _mm256_set1_epi32(0); return tmp; }
    static INLINE TV pack_unordered(TV, TV) { TV tmp = _mm256_set1_epi32(0); return tmp; }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};
