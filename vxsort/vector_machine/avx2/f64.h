template <>
class vxsort_machine_traits<f64, AVX2> {
public:
    typedef f64 T;
    typedef __m256d TV;
    typedef __m256i TLOADSTOREMASK;
    typedef u32 TMASK;
    typedef f64 TPACK;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return false; }
    static constexpr bool supports_packing() { return false; }

    template <i32 Shift>
    static bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(prefix_mask_table_64b + amount * N)));
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(suffix_mask_table_64b + amount * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_loadu_pd((T *)p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_pd((T *)ptr, v); }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_partial_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return i2d(_mm256_or_si256(d2i(_mm256_maskload_pd((T *) p, mask)),
                                   _mm256_andnot_si256(mask, d2i(base))));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_pd((double *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, i32 mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2d(_mm256_permutevar8x32_ps(d2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }

    static INLINE TV broadcast(T pivot) { return _mm256_set1_pd(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) {
        ///    0x0E: Greater-than (ordered, signaling) \n
        ///    0x1E: Greater-than (ordered, non-signaling)
        return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OS));
    }

    static TV shift_right(TV v, i32 i) { return v; }
    static TV shift_left(TV v, i32 i) { return v; }

    static INLINE TV add(TV a, TV b) { return _mm256_add_pd(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_pd(a, b); };

    static INLINE TV pack_unordered(TV, TV) { TV tmp = _mm256_set1_pd(0); return tmp; }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <i32 Shift>
    static T shift_n_sub(T v, T sub) { return v; }

    template <i32 Shift>
    static T unshift_and_add(TPACK from, T add) { return add; }
};
