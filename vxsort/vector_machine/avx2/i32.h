template <>
class vxsort_machine_traits<i32, AVX2> {
public:
    typedef i32 T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef u32 TMASK;
    typedef i16 TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return false; }
    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T span) {
        const auto PACK_LIMIT = (((TU)std::numeric_limits<u16>::max() + 1)) << Shift;
        return ((TU)span) < PACK_LIMIT;
    }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount < N);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(prefix_mask_table_32b + N * amount)));
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount < N);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(suffix_mask_table_32b + N * amount)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi32((i32 *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) { _mm256_maskstore_epi32((i32 *) p, mask, v); }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE TV broadcast(i32 pivot) { return _mm256_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(a, b))); }

    static TV shift_right(TV v, int i) { return _mm256_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm256_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi32(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) {
        auto shifted = _mm256_slli_epi32(b, 16);
        return _mm256_blend_epi16(a, shifted, 0b10101010);
    }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        auto p01 = _mm256_extracti128_si256(p, 0);
        auto p02 = _mm256_extracti128_si256(p, 1);

        u1 = _mm256_cvtepi16_epi32(p01);
        u2 = _mm256_cvtepi16_epi32(p02);
    }

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
