#!/bin/sh

# Source:
# avx3 check follows: https://github.com/datastax/jvector/blob/4f13ff42e93f2e7b16020de3aaeefe9372e0dd36/jvector-native/src/main/native/tests/test_x86_cpu_features.cpp#L51-L56
# avx3_spr partially follows: https://github.com/datastax/jvector/blob/4f13ff42e93f2e7b16020de3aaeefe9372e0dd36/jvector-native/src/main/native/tests/test_x86_cpu_features.cpp#L68-L71

get_avx_support() {
  if lscpu | grep -i avx512f | grep -i avx512bw | grep -i avx512cd | grep -i avx512dq | grep -i avx512vl; then
    if lscpu | grep -q "GenuineIntel" && lscpu | grep -i avx512_fp16 | grep -i avx512_bf16 | grep -i avx512_vpopcntdq; then
      echo "AVX3_SPR"
    else
      echo "AVX3"
    fi
  elif lscpu | grep -qi avx2; then
    echo "AVX2"
  else
    echo "NO_AVX"
  fi
}

print_avx_support_message() {
  avx_support=$1
  if [ "$avx_support" = "AVX3_SPR" ]; then
    echo "the system is an Intel(R) Sapphire Rapids or a newer-generation processor"
  elif [ "$avx_support" = "AVX3" ]; then
    echo "avx512 available on system"
  elif [ "$avx_support" = "AVX2" ]; then
    echo "avx2 available on system"
  else
    echo "avx512 and avx2 not available on system"
  fi
}

check_avx_support() {
  avx_support=$(get_avx_support)
  print_avx_support_message "$avx_support"
  echo "$avx_support"
}

get_avx_build_flags() {
  avx_support=$1
  if [ "$avx_support" = "AVX3_SPR" ]; then
    echo "-Davx512_spr.enabled=true -Davx512.enabled=false"
  elif [ "$avx_support" = "AVX3" ]; then
    echo "-Davx512_spr.enabled=false -Davx512.enabled=true"
  elif [ "$avx_support" = "AVX2" ]; then
    echo "-Davx512.enabled=false -Davx512_spr.enabled=false"
  else
    echo "-Davx2.enabled=false -Davx512.enabled=false -Davx512_spr.enabled=false -Dsimd.enabled=false"
  fi
}
