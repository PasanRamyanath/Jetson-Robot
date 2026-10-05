idf_component_register(
  SRCS "main.c" "motor.c" "sensors.c" "vl53l0x.c"
  INCLUDE_DIRS "." "${CMAKE_CURRENT_LIST_DIR}/../../../shared/proto"
  PRIV_REQUIRES driver esp_adc esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Wno-unused-parameter)
