template <>
class vxsort_machine_traits<double, AVX512> {
public:
    typedef double T;
    typedef __m512d TV;
    typedef __mmask8 TLOADSTOREMASK;
    typedef __mmask8 TMASK;
    typedef double TPACK;

    static const int N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return true; }
    static constexpr bool type_supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder <= 8);
        return 0xFF >> ((N - remainder) & (N-1));
    }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_pd(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_pd(ptr, v); }

    static INLINE TV load_masked_vec(TV *ptr, TV base, TLOADSTOREMASK mask) {
        return _mm512_mask_loadu_pd(base, mask, (T const *) ptr);
    }

    static INLINE void store_masked_vec(TV * p, TV v, TLOADSTOREMASK mask) {
        _mm512_mask_storeu_pd(p, mask, v);
    }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_pd(ptr, mask, v); }
    static INLINE TV broadcast(double pivot) { return _mm512_set1_pd(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS); }

    static TV shift_right(TV v, int i) { return v; }
    static TV shift_left(TV v, int i) { return v; }

    static INLINE TV add(TV a, TV b) { return _mm512_add_pd(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_pd(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) { return a; }
    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) { }
    template <int Shift>
    static T shift_n_sub(T v, T sub) { return v; }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) { return add; }
};