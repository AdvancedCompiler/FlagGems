add_executable(test_triton_pointwise test_triton_pointwise.cpp)
target_link_libraries(test_triton_pointwise
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_pointwise COMMAND test_triton_pointwise)

add_executable(test_triton_reduction test_triton_reduction.cpp)
target_link_libraries(test_triton_reduction
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_reduction COMMAND test_triton_reduction)

add_executable(test_triton_blas test_triton_blas.cpp)
target_link_libraries(test_triton_blas PRIVATE Torch::Torch operators gtest gtest_main)
add_test(NAME test_triton_blas COMMAND test_triton_blas)

add_executable(test_triton_norm test_triton_norm.cpp)
target_link_libraries(test_triton_norm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_norm COMMAND test_triton_norm)

add_executable(test_triton_rope test_triton_rope.cpp)
target_link_libraries(test_triton_rope
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_rope COMMAND test_triton_rope)


add_executable(test_triton_topk test_triton_topk.cpp)
target_link_libraries(test_triton_topk
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_topk COMMAND test_triton_topk)

add_executable(test_triton_contiguous test_triton_contiguous.cpp)
target_link_libraries(test_triton_contiguous
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_contiguous COMMAND test_triton_contiguous)

add_executable(test_triton_cat test_triton_cat.cpp)
target_link_libraries(test_triton_cat
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_cat COMMAND test_triton_cat)

add_executable(test_triton_zeros_constructor test_triton_tensor_constructor.cpp)
target_link_libraries(test_triton_zeros_constructor
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_zeros_constructor COMMAND test_triton_zeros_constructor)

add_executable(test_triton_bmm test_triton_bmm.cpp)
target_link_libraries(test_triton_bmm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_bmm COMMAND test_triton_bmm)

add_executable(test_triton_embedding test_triton_embedding.cpp)
target_link_libraries(test_triton_embedding
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_embedding COMMAND test_triton_embedding)

add_executable(test_triton_argmax test_triton_argmax.cpp)
target_link_libraries(test_triton_argmax
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_argmax COMMAND test_triton_argmax)

add_executable(test_triton_fill  test_triton_fill.cpp)
target_link_libraries(test_triton_fill
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_fill COMMAND test_triton_fill)
