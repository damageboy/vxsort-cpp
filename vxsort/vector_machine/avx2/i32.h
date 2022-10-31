template <>
class vxsort_machine_traits<i32, AVX2> {
public:
    typedef i32 T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef u32 TCMPMASK;
    typedef i16 TPACK;
    typedef typename std::make_unsigned<T>::type TU;
    static_assert(sizeof(TPACK)*2 == sizeof(T), "TPACK must be half-width of T");

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return false; }
    static constexpr bool supports_packing() { return false; }

    template <i32 Shift>
    static bool can_pack(T span) {
        constexpr auto PACK_LIMIT = (((TU)std::numeric_limits<u16>::max() + 1)) << Shift;
        return ((TU)span) < PACK_LIMIT;
    }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(prefix_mask_table_32b + N * amount)));
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(suffix_mask_table_32b + N * amount)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static INLINE void store_compress_vec(TV*, TV, TCMPMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_partial_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi32((i32 *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) { _mm256_maskstore_epi32((i32 *) p, mask, v); }

    static INLINE TV partition_vector(TV v, i32 mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE TV broadcast(T pivot) { return _mm256_set1_epi32(pivot); }

    static INLINE TCMPMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(a, b))); }

    static INLINE TV shift_right(TV v, i32 i) { return _mm256_srli_epi32(v, i); }
    static INLINE TV shift_left(TV v, i32 i) { return _mm256_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi32(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) { return _mm256_packs_epi32(a, b); }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        u1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p, 0));
        u2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p, 1));
    }

    template <i32 Shift>
    static INLINE T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <i32 Shift>
    static INLINE T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};
