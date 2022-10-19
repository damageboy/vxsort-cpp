template <>
class vxsort_machine_traits<i64, AVX512> {
public:
    typedef i64 T;
    typedef __m512i TV;
    typedef __mmask8 TLOADSTOREMASK;
    typedef __mmask8 TMASK;
    typedef i32 TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return true; }
    static constexpr bool supports_packing() { return true; }

    template <i32 Shift>
    static bool can_pack(T span) {
        constexpr auto PACK_LIMIT = (((TU) std::numeric_limits<u32>::max() + 1)) << Shift;
        return ((TU) span) < PACK_LIMIT;
    }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return 0xFF >> ((N - amount) & (N-1));
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);
        return 0xFF << (amount & (N-1));
    }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    static INLINE TV load_partial_vec(TV *ptr, TV base, TLOADSTOREMASK mask) {
        return _mm512_mask_loadu_epi64(base, mask, (i64 const *) ptr);
    }

    static INLINE void store_masked_vec(TV * p, TV v, TLOADSTOREMASK mask) {
        _mm512_mask_storeu_epi64(p, mask, v);
    }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi64(ptr, mask, v); }

    static INLINE TV broadcast(T pivot) { return _mm512_set1_epi64(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, i32 i) { return _mm512_srli_epi64(v, i); }
    static TV shift_left(TV v, i32 i) { return _mm512_slli_epi64(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi64(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi64(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) { return _mm512_mask_shuffle_epi32(a, 0b1010101010101010, b, _MM_PERM_CDAB); }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        auto p01 = _mm512_extracti32x8_epi32(p, 0);
        auto p02 = _mm512_extracti32x8_epi32(p, 1);

        u1 = _mm512_cvtepi32_epi64(p01);
        u2 = _mm512_cvtepi32_epi64(p02);
    }

    template <i32 Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <i32 Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;

        if (Shift > 0)
            add = (T) (((TU) add) << Shift);

        return add;
    }

};