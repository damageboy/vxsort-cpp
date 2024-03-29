template <>
class vxsort_machine_traits<f32, AVX512> {
public:
    typedef f32 T;
    typedef __m512 TV;
    typedef __mmask16 TLOADSTOREMASK;
    typedef __mmask16 TCMPMASK;
    typedef f32 TPACK;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return true; }
    static constexpr bool supports_packing() { return false; }

    template <i32 Shift>
    static bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return 0xFFFF >> ((N - amount) & (N-1));
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return 0xFFFF << (amount & (N-1));
    }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_ps(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_ps(ptr, v); }

    static TV load_partial_vec(TV *ptr, TV base, TLOADSTOREMASK mask) {
        return _mm512_mask_loadu_ps(base, mask, (T const *) ptr);
    }

    static INLINE void store_masked_vec(TV * p, TV v, TLOADSTOREMASK mask) {
        _mm512_mask_storeu_ps(p, mask, v);
    }

    // Will never be called
    static INLINE TV partition_vector(TV v, i32 mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TCMPMASK mask) { _mm512_mask_compressstoreu_ps(ptr, mask, v); }

    static INLINE TV broadcast(T pivot) { return _mm512_set1_ps(pivot); }

    static INLINE TCMPMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS); }

    static INLINE TV shift_right(TV v, i32 i) { return v; }
    static INLINE TV shift_left(TV v, i32 i) { return v; }

    static INLINE TV add(TV a, TV b) { return _mm512_add_ps(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_ps(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) { return a; }
    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) { }
    template <i32 Shift>
    static INLINE T shift_n_sub(T v, T sub) { return v; }

    template <i32 Shift>
    static INLINE T unshift_and_add(TPACK from, T add) { return add; }
};
