template <>
class vxsort_machine_traits<i32, AVX512> {
public:
    typedef i32 T;
    typedef __m512i TV;
    typedef __mmask16 TLOADSTOREMASK;
    typedef __mmask16 TMASK;
    typedef i16 TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return true; }
    static constexpr bool supports_packing() { return true; }

    template <i32 Shift>
    static bool can_pack(T span) {
        if (!supports_vector_machine<AVX512>(16)) {
            return false;
        }
        constexpr auto PACK_LIMIT = (((TU)std::numeric_limits<u16>::max() + 1)) << Shift;
        return ((TU)span) < PACK_LIMIT;
    }

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

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    static TV load_partial_vec(TV* ptr, TV base, TLOADSTOREMASK mask) { return _mm512_mask_loadu_epi32(base, mask, (i32 const*)ptr); }

    static INLINE void store_masked_vec(TV* p, TV v, TLOADSTOREMASK mask) { _mm512_mask_storeu_epi32(p, mask, v); }

    // Will never be called
    static INLINE TV partition_vector(TV v, i32 mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi32(ptr, mask, v); }

    static INLINE TV broadcast(T pivot) { return _mm512_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT); }

    static INLINE TV shift_right(TV v, i32 i) { return _mm512_srli_epi32(v, i); }
    static INLINE TV shift_left(TV v, i32 i) { return _mm512_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi32(a, b); };

    static INLINE TV pack_unordered(TV a, TV b) { return _mm512_packs_epi32(a, b); }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        u1 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(p, 0));
        u2 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(p, 1));
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
